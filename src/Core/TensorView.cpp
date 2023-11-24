#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/Lower.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Size.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Utils/Common.hpp"

namespace kas {

std::string PureTensor::shapeToString(const BindingContext& ctx) const {
    return getShape().toString(ctx);
}

std::string PureTensor::description(const BindingContext& ctx) const {
    return content.description(ctx);
}

std::string AbstractAccess::outerLoopsIteratorsToString() const {
    return VectorToString(outerLoops
        | std::views::transform([](const IteratorValue& it) -> const std::string& {
            return it.as<VariableValueNode>().name;
        })
    );
}

std::string AbstractAccess::innerLoopsIteratorsToString() const {
    return VectorToString(innerLoops
        | std::views::transform([](const IteratorValue& it) -> const std::string& {
            return it.as<VariableValueNode>().name;
        })
    );
}

std::string AbstractAccess::accessToString(const BindingContext& ctx, int pos) const {
    return VectorToString((pos == TensorExpression::Output ? output : inputs.at(pos))
        | std::views::transform([&](const IteratorValue& it) {
            return it.toString(ctx);
        })
    );
}

namespace {

class ConcreteTensorExpressionPrinter final: public TensorExpressionPrinter {
    const BindingContext& ctx;
    const AbstractAccess& access;
    bool isOutput(const TensorTensorExpression& tensor) const {
        return tensor.position == TensorExpression::Output;
    }
public:
    ConcreteTensorExpressionPrinter(const BindingContext& ctx, const AbstractAccess& access):
        ctx { ctx },
        access { access }
    {}
    void visit(TensorTensorExpression& expr) override {
        if (access.isDerivative() && isOutput(expr)) {
            ss << "grad_"; // This is the derivative of the output.
        }
        TensorExpressionPrinter::visit(expr); // print the tensor name
        ss << access.accessToString(ctx, expr.position); // print the access iterators
    };
    std::string print(const TensorExpression& originalExpr) {
        TensorExpression expr = originalExpr;
        if (access.isDerivative()) {
            expr *= TensorTensorExpression::Create(TensorExpression::Output);
        }
        if (access.divBy.has_value()) {
            ss << "(";
        }
        expr.accept(*this);
        if (access.divBy.has_value()) {
            ss << ") / (" << access.divBy->toString(ctx) << ")";
        }
        std::string result = ss.str();
        ss.str("");
        return result;
    }
};

} // namespace

std::string AbstractAccess::statementToString(const BindingContext& ctx) const {
    ConcreteTensorExpressionPrinter printer { ctx, *this };
    return printer.print(expression);
}

std::string AbstractAccess::targetEntryToString() const {
    if (position == TensorExpression::Output) {
        return fmt::format("out{}", outerLoopsIteratorsToString());
    } else {
        return fmt::format("grad_in_{}{}", position, outerLoopsIteratorsToString());
    }
}

ConcreteConsts TensorView::computePadding(const BindingContext& ctx, const Graph& graph, const ConcreteConsts& consts) const {
    PaddingSolver sol { ctx, consts };
    // Find all merges.
    for (const MergeLikeOp *op: graph.getOpsOfType<MergeLikeOp>(DimensionType::Merge)) {
        sol.addConstraint(op->getInputL().size());
        sol.addConstraint(op->getInputR().size());
    }
    for (auto it: interface) {
        sol.addConstraint(it->size());
    }
    for (auto it: manipulations) {
        sol.addConstraint(it->size());
    }
    return sol.solve(ShapeView(getUnderlyingTensors().at(0).getDims()).totalSize(), Size::Product(getInterfaceShape()));
}

std::size_t TensorView::getFLOPs(const BindingContext& ctx, const ConcreteConsts& consts) const {
    return getSubgraphs().getFLOPs(ctx, consts);
}
std::size_t TensorView::getFLOPs(const BindingContext& ctx) const {
    return getSubgraphs().getFLOPs(ctx);
}

namespace {
    auto SSIt(std::stringstream& ss) -> decltype(auto) {
        return std::ostreambuf_iterator<char>(ss);
    }
    void IndentSpaces(std::stringstream& ss, std::size_t indent) {
        fmt::format_to(SSIt(ss), "{:{}}", "", 4 * indent);
    }
}

std::string TensorView::printNestedLoops(const BindingContext& ctx, int pos) const {
    std::stringstream ss;
    std::size_t depth = 0;

    auto& access = pos == TensorExpression::Output ? forwardAccess : backwardAccesses.at(pos);

    for (auto it: access.outerLoops) {
        const auto& var = it.as<VariableValueNode>();
        IndentSpaces(ss, depth);
        fmt::format_to(SSIt(ss),
            "for (int {0} = 0; {0} < {1}; {0}++) {{\n",
            var.name,
            (pos == TensorExpression::Output ? interface.at(depth)->size() : tensors.at(pos).getShape()[depth]).toString(ctx)
        );
        ++depth;
    }

    const auto recursion = [&ctx, &ss, &access](const auto& self, std::size_t innerIndex, std::size_t depth) -> std::string {
        if (innerIndex-- == 0) {
            return access.statementToString(ctx);
        }

        auto& m = access.innerLoops.at(innerIndex).as<VariableValueNode>();
        std::string tempName = "temp_" + m.name;

        // float temp_ri_idx = 0;
        IndentSpaces(ss, depth);
        fmt::format_to(SSIt(ss), "float {} = 0;\n", tempName);

        // for (int ri_idx = 0; ri_idx < size_ri_idx; ri_idx++) {
        IndentSpaces(ss, depth);
        fmt::format_to(SSIt(ss), "for (int {0} = 0; {0} < {1}; {0}++) {{\n", m.name, access.innerLoopsShape[innerIndex].toString(ctx));

        // Generate inner loops, and obtain the temporary variable.
        std::string inner = self(self, innerIndex, depth + 1);

        IndentSpaces(ss, depth + 1);
        fmt::format_to(SSIt(ss), "{} += {};\n", tempName, inner);

        IndentSpaces(ss, depth);
        ss << "}\n";

        return tempName;
    };
    std::string lastTemp = recursion(recursion, access.innerLoops.size(), depth);
    IndentSpaces(ss, depth);
    fmt::format_to(SSIt(ss), "{} = {};\n", access.targetEntryToString(), lastTemp);

    while (depth --> 0) {
        IndentSpaces(ss, depth);
        ss << "}\n";
    }
    return ss.str();
}

std::string TensorView::printNestedLoopsForAll(const BindingContext& ctx) const {
    std::stringstream ss;
    for (int i = TensorExpression::Output; i < static_cast<int>(tensors.size()); ++i) {
        fmt::format_to(std::ostreambuf_iterator<char>(ss), "/* Loops {}: {} */\n", i, i == TensorExpression::Output ? "Forward Kernel" : fmt::format("Backward Kernel for Input {}", i));
        ss << printNestedLoops(ctx, i);
    }
    return ss.str();
}

std::string TensorView::description(const BindingContext& ctx) const {
    return Topmost::Description(tensors | std::views::transform(&PureTensor::getContent), ctx);
}

TensorView::TensorView(const std::vector<Topmost>& canonicalTensors, TensorExpression blending, const BindingContext& ctx) {
    subgraphs = IR::Build(canonicalTensors, ctx);
    KAS_ASSERT(subgraphs.inputTensors.at(0).output() == canonicalTensors.at(0).getDimensions());

    const Graph graph = subgraphs.buildGraph();

    const auto& outputIterators = graph.getOutputIterators();
    const auto& reduceIterators = graph.getReduceIterators();

    for (std::size_t tId = 0; const auto& tensor: subgraphs.inputTensors) {
        this->tensors.emplace_back(tId, tensor.output(), subgraphs.expansions.at(tId));
        ++tId;
    }

    std::vector<Dimension> interfaceDimensions;
    std::ranges::copy(outputIterators, std::back_inserter(interfaceDimensions));

    auto forwardEval = DimensionEvaluator(graph, this->tensors, blending);
    forwardEval.makeVars(interfaceDimensions);
    for (auto r: reduceIterators) {
        forwardEval.reduceAt(r);
    }
    forwardEval.adjustReductionOrder();
    forwardAccess = forwardEval.toAccess(TensorExpression::Output, interfaceDimensions);
    for (std::size_t tId = 0; auto&& tensor: this->tensors) {
        // KAS_DEBUG("Differentiating input {}...", tId);
        auto backwardEval = DimensionEvaluator(graph, this->tensors, blending);
        backwardEval.makeVars(tensor.getDims());
        backwardEval.fillWithReductions();
        backwardEval.adjustReductionOrder();
        backwardAccesses.emplace_back(backwardEval.toAccess(tId, interfaceDimensions));
        ++tId;
    }

    interface = outputIterators;
    manipulations = reduceIterators;
}

} // namespace kas
