#include "KAS/Core/Lower.hpp"
#include "KAS/Utils/Algorithm.hpp"


namespace kas {

void DimensionEvaluator::assign(const Dimension& dim, IteratorValue value) {
    Valuation& val = values[dim];
    KAS_ASSERT(val.type() == Valuation::Type::Unoriented, "Assigning to assigned or constrained dimension.");
    val = Valuation { value };
    freeCandidates.erase(dim);
    unknownDimensions.erase(dim);
    Propagator p { *this };
    graph.visitAlong(dim, Direction::Up).match(p, p, p);
    graph.visitAlong(dim, Direction::Down).match(p, p, p);
}

DimensionEvaluator::DimensionEvaluator(const Graph& graph):
    graph { graph }
{
    // Collect expression info from graph.
    auto avg = graph.getMapReduceIterators()
        | std::views::filter([](const MapReduceOp *op) {
            return op->getReduce() == MapReduceOp::ReduceType::Mean;
        })
        | std::views::transform(&MapReduceOp::size);
    divBy = FoldLeftFirst(avg, std::multiplies<Size>{});

    // Initialize the unknown and free set.
    std::ranges::copy(graph.getDimensions(), std::inserter(unknownDimensions, unknownDimensions.end()));
    std::ranges::copy(graph.getDimensions(), std::inserter(freeCandidates, freeCandidates.end()));

    // Perform initial visit to each Op. This is needed to handle StrideOp, which comes with an initial Orientation.
    Propagator p { *this };
    for (auto&& dim: graph.getDimensions()) {
        graph.visitAlong(dim, Direction::Down).match(p, p, p);
    }
}

void DimensionEvaluator::makeVar(const Dimension& dim) {
    outerLoops.emplace_back(VariableValueNode::Create(false, outerLoops.size(), "i_" + std::to_string(outerLoops.size())));
    assign(dim, outerLoops.back());
}

void DimensionEvaluator::makeVars(const std::vector<Dimension>& dims) {
    for (const auto& dim: dims) {
        makeVar(dim);
    }
}

void DimensionEvaluator::reduceAt(const Dimension& dim) {
    innerLoopsShape.emplace_back(dim.size());
    innerLoops.emplace_back(VariableValueNode::Create(true, innerLoops.size(), "ri_" + std::to_string(innerLoops.size())));
    assign(dim, innerLoops.back());
}

void DimensionEvaluator::fillWithReductions() {
    while (!unknownDimensions.empty()) {
        KAS_ASSERT(!freeCandidates.empty(), "No free candidates left.");
        Dimension free = *freeCandidates.begin();
        reduceAt(free);
    }
}

std::vector<IteratorValue> DimensionEvaluator::extractValues(const std::vector<Dimension>& dims) const {
    KAS_ASSERT(unknownDimensions.empty());
    std::vector<IteratorValue> result;
    std::ranges::move(dims | std::views::transform([&](const auto& dim) { return values.at(dim).extractValue(); }), std::back_inserter(result));
    return result;
}

AbstractAccess DimensionEvaluator::toAccess(int position, const std::vector<PureTensor>& inputs, const std::vector<Dimension>& output) {
    std::vector<std::vector<IteratorValue>> inputsAccesses;
    std::ranges::move(inputs | std::views::transform([&](const auto& tensor) { return extractValues(tensor.getDimensions()); }), std::back_inserter(inputsAccesses));
    return AbstractAccess {
        .position = position,
        .outerLoops = std::move(outerLoops),
        .innerLoops = std::move(innerLoops),
        .innerLoopsShape = std::move(innerLoopsShape),
        .inputs = std::move(inputsAccesses),
        .output = extractValues(output),
        .divBy = std::move(divBy)
    };
}

} // namespace kas
