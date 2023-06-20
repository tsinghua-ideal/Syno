#include <limits>

#include "KAS/Core/Lower.hpp"
#include "KAS/Utils/Algorithm.hpp"


namespace kas {

TensorExpression TensorExpressionDifferentiator::visits(IntegerTensorExpression& expr) {
    return IntegerTensorExpression::Create(0);
}
TensorExpression TensorExpressionDifferentiator::visits(TensorTensorExpression& expr) {
    if (position == expr.position) {
        return IntegerTensorExpression::Create(1);
    } else {
        return IntegerTensorExpression::Create(0);
    }
}
TensorExpression TensorExpressionDifferentiator::visits(BinaryOpTensorExpression& expr) {
    switch (expr.op) {
    case BinaryOpTensorExpression::Op::Add:
        return differentiate(expr.lhs) + differentiate(expr.rhs);
    case BinaryOpTensorExpression::Op::Mul:
        return differentiate(expr.lhs) * expr.rhs + expr.lhs * differentiate(expr.rhs);
    default:
        KAS_UNREACHABLE();
    }
}
TensorExpression TensorExpressionDifferentiator::differentiate(TensorExpression& expr) {
    expr.accept(*this);
    return result();
}

void DimensionEvaluator::assign(Dimension dim, IteratorValue value) {
    Valuation& val = values[dim];
    KAS_ASSERT(val.type() == Valuation::Type::Unoriented, "Assigning to assigned or constrained dimension.");
    val = Valuation { value };
    freeCandidates.erase(dim);
    unknownDimensions.erase(dim);
    Propagator p { *this };
    graph.visitAlong(dim, Direction::Up).match(p, p, p);
    graph.visitAlong(dim, Direction::Down).match(p, p, p);
}

std::vector<IteratorValue> DimensionEvaluator::extractValues(const std::vector<Dimension>& dims) const {
    KAS_ASSERT(unknownDimensions.empty());
    std::vector<IteratorValue> result;
    std::ranges::move(dims | std::views::transform([&](const auto& dim) { return values.at(dim).extractValue(); }), std::back_inserter(result));
    return result;
}

DimensionEvaluator::DimensionEvaluator(const Graph& graph, const std::vector<PureTensor>& inputTensors):
    graph { graph },
    inputTensors { inputTensors }
{
    // Collect expression info from graph.
    auto avg = graph.getMapReduceIterators()
        | std::views::filter([](const MapReduce *op) {
            return op->getReduce() == MapReduce::ReduceType::Mean;
        })
        | std::views::transform(&MapReduce::size);
    divBy = FoldLeftFirst(avg, std::multiplies<Size>{});

    // Initialize the unknown and free set.
    std::ranges::copy(graph.getDimensions(), std::inserter(unknownDimensions, unknownDimensions.end()));
    std::ranges::copy(graph.getDimensions(), std::inserter(freeCandidates, freeCandidates.end()));

    // Perform initial visit to each Op. This is needed to handle StrideOp, which comes with an initial Orientation.
    Propagator p { *this };
    for (auto&& dim: graph.getDimensions()) {
        graph.visitAlong(dim, Direction::Down).match(p, p, p);
    }

    // TODO!!! Accept expression from user.
    expression = *FoldLeftFirst(std::views::iota(static_cast<std::size_t>(0), inputTensors.size()) | std::views::transform(&TensorTensorExpression::Create), std::multiplies<>{});
}

void DimensionEvaluator::makeVar(Dimension dim) {
    outerLoops.emplace_back(VariableValueNode::Create(false, outerLoops.size(), "i_" + std::to_string(outerLoops.size())));
    assign(dim, outerLoops.back());
}

void DimensionEvaluator::makeVars(const std::vector<Dimension>& dims) {
    for (const auto& dim: dims) {
        makeVar(dim);
    }
}

void DimensionEvaluator::reduceAt(Dimension dim) {
    innerLoopsShape.emplace_back(dim.size());
    innerLoops.emplace_back(VariableValueNode::Create(true, innerLoops.size(), "unadjusted_ri_" + std::to_string(innerLoops.size())));
    assign(dim, innerLoops.back());
}

void DimensionEvaluator::fillWithReductions() {
    auto attemptToReduceInputDimension = [&]() -> bool {
        // We want to speed up backward pipeline.
        // (Note that forward pipeline does not even need to call this.)
        // Use input dimensions as reduction variables. In this way we can increase probability of loop fusion. And we can make better use of locality as well.
        for (const auto& inputTensor: inputTensors) {
            for (const Dimension& inputDim: inputTensor.getDimensions() | std::views::reverse) {
                // In reverse. Because we use row-major representation, and it is better to reduce the innermost dimension first.
                if (freeCandidates.contains(inputDim)) {
                    reduceAt(inputDim);
                    return true;
                }
            }
        }
        return false;
    };
    while (!unknownDimensions.empty()) {
        KAS_ASSERT(!freeCandidates.empty(), "No free candidates left.");
        bool success = attemptToReduceInputDimension();
        if (!success) {
            // If we cannot reduce any input dimension, then we just reduce any free dimension.
            ++CountCannotFindInputDimensionToReduce;
            reduceAt(*freeCandidates.begin());
        }
    }
}

void DimensionEvaluator::adjustReductionOrder() {
    KAS_ASSERT(unknownDimensions.empty());
    // From inner loops to outer loops.
    const std::size_t totalLoops = innerLoops.size();
    constexpr std::size_t remainTheSame = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> transposition(totalLoops, remainTheSame);
    std::size_t adjusted = 0;
    for (const auto& inputTensor: inputTensors) {
        for (const Dimension& inputDim: inputTensor.getDimensions() | std::views::reverse) {
            if (auto var = values.at(inputDim).extractValue().tryAs<VariableValueNode>(); var && var->isReduce) {
                if (transposition[var->index] == remainTheSame) {
                    transposition[var->index] = adjusted++;
                }
            }
        }
    }
    // We want the reductions not in inputs just remain the same order, and prioritize them.
    // The new order is: [other reductions] + [reductions in inputs].
    const std::size_t delta = totalLoops - adjusted;
    std::size_t unadjusted = 0;
    for (std::size_t from = 0; from < totalLoops; ++from) {
        if (transposition[from] == remainTheSame) {
            transposition[from] = unadjusted++;
        } else {
            transposition[from] += delta;
        }
    }
    KAS_ASSERT(unadjusted + adjusted == totalLoops);

    // Now we have obtained the transposition. Apply it.
    std::vector<IteratorValue> newInnerLoops(totalLoops);
    std::vector<std::size_t> inverseTransposition(totalLoops, remainTheSame);
    for (std::size_t fro = 0; fro < totalLoops; ++fro) {
        const std::size_t to = transposition[fro];
        auto& var = innerLoops[fro].as<VariableValueNode>();
        var.index = to;
        var.name = "ri_" + std::to_string(to);
        newInnerLoops[to] = std::move(innerLoops[fro]);
        inverseTransposition[to] = fro;
    }
    innerLoops = std::move(newInnerLoops);
    std::vector<Size> newInnerLoopsShape;
    for (std::size_t to = 0; to < totalLoops; ++to) {
        newInnerLoopsShape.emplace_back(innerLoopsShape.at(inverseTransposition[to]));
    }
    innerLoopsShape = std::move(newInnerLoopsShape);
}

AbstractAccess DimensionEvaluator::toAccess(int position, const std::vector<Dimension>& output) {
    std::vector<std::vector<IteratorValue>> inputsAccesses;
    std::ranges::move(inputTensors | std::views::transform([&](const auto& tensor) { return extractValues(tensor.getDimensions()); }), std::back_inserter(inputsAccesses));
    return AbstractAccess {
        .position = position,
        .outerLoops = std::move(outerLoops),
        .innerLoops = std::move(innerLoops),
        .innerLoopsShape = std::move(innerLoopsShape),
        .inputs = std::move(inputsAccesses),
        .output = extractValues(output),
        .expression =
            position == TensorExpression::Output ?
            expression :
            TensorExpressionDifferentiator(position).differentiate(expression),
        .divBy = std::move(divBy),
    };
}

} // namespace kas
