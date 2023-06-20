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
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Size.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Common.hpp"

namespace kas {

std::string PureTensor::shapeToString(const BindingContext& ctx) const {
    return getShape().toString(ctx);
}

std::string PureTensor::description(const BindingContext& ctx) const {
    return DimensionArrayToString(dims, ctx);
}

TensorExpression TensorExpression::operator+(const TensorExpression& other) const {
    return BinaryOpTensorExpression::Create(*this, other, BinaryOpTensorExpression::Op::Add);
}
TensorExpression& TensorExpression::operator+=(const TensorExpression& other) {
    return *this = *this + other;
}
TensorExpression TensorExpression::operator*(const TensorExpression& other) const {
    return BinaryOpTensorExpression::Create(*this, other, BinaryOpTensorExpression::Op::Mul);
}
TensorExpression& TensorExpression::operator*=(const TensorExpression& other) {
    return *this = *this * other;
}

TensorExpression IntegerTensorExpression::Create(int value) {
    static auto zero = TensorExpression(std::make_shared<IntegerTensorExpression>(0));
    static auto one = TensorExpression(std::make_shared<IntegerTensorExpression>(1));
    if (value == 0) {
        return zero;
    } else if (value == 1) {
        return one;
    }
    return TensorExpression(std::make_shared<IntegerTensorExpression>(value));
}

std::string TensorTensorExpression::toString() const {
    if (position == TensorExpression::Output) {
        return "out";
    } else {
        return "in_" + std::to_string(position);
    }
}

std::string BinaryOpTensorExpression::toString() const {
    switch (op) {
    case Op::Add:
        return fmt::format("({}) + ({})", lhs.toString(), rhs.toString());
    case Op::Mul:
        return fmt::format("({}) * ({})", lhs.toString(), rhs.toString());
    default:
        KAS_UNREACHABLE();
    }
}

TensorExpression BinaryOpTensorExpression::Create(TensorExpression lhs, TensorExpression rhs, Op op) {
    switch (op) {
    case Op::Add:
        if (auto lhsInt = lhs.tryAs<IntegerTensorExpression>(); lhsInt) {
            if (lhsInt->value == 0) {
                return rhs;
            }
            if (auto rhsInt = rhs.tryAs<IntegerTensorExpression>(); rhsInt) {
                return TensorExpression(std::make_shared<IntegerTensorExpression>(lhsInt->value + rhsInt->value));
            }
        } else if (auto rhsInt = rhs.tryAs<IntegerTensorExpression>(); rhsInt) {
            if (rhsInt->value == 0) {
                return lhs;
            }
        }
        break;
    case Op::Mul:
        if (auto lhsInt = lhs.tryAs<IntegerTensorExpression>(); lhsInt) {
            if (lhsInt->value == 0) {
                return lhs;
            } else if (lhsInt->value == 1) {
                return rhs;
            }
            if (auto rhsInt = rhs.tryAs<IntegerTensorExpression>(); rhsInt) {
                return TensorExpression(std::make_shared<IntegerTensorExpression>(lhsInt->value * rhsInt->value));
            }
        } else if (auto rhsInt = rhs.tryAs<IntegerTensorExpression>(); rhsInt) {
            if (rhsInt->value == 0) {
                return rhs;
            } else if (rhsInt->value == 1) {
                return lhs;
            }
        }
        break;
    }
    return TensorExpression(std::make_shared<BinaryOpTensorExpression>(std::move(lhs), std::move(rhs), op));
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

std::string AbstractAccess::statementToString(const BindingContext& ctx) const {
    return fmt::format("{}{}", fmt::join(
        std::views::iota(std::size_t{0}, inputs.size())
        | std::views::transform([&](std::size_t pos) {
            if (pos == position) {
                return fmt::format("grad_out{}", accessToString(ctx, TensorExpression::Output));
            } else {
                return fmt::format("in_{}{}", pos, accessToString(ctx, pos));
            }
        }),
    " * "),
    divBy ? fmt::format(" / ({})", divBy->toString(ctx)) : ""
    );
}

std::string AbstractAccess::targetEntryToString() const {
    if (position == TensorExpression::Output) {
        return fmt::format("out{}", outerLoopsIteratorsToString());
    } else {
        return fmt::format("grad_in_{}{}", position, outerLoopsIteratorsToString());
    }
}

ConcreteConsts TensorView::computePadding(const BindingContext& ctx, const ConcreteConsts& consts) const {
    PaddingSolver sol { ctx, consts };
    // Find all merges.
    struct visitor: public DimVisitor {
        PaddingSolver& sol;
        std::set<Dimension, Dimension::AddressLessThan> visited;
        visitor(PaddingSolver& sol): sol(sol) {}
        void visit(const RepeatLikeOp::Input& dim) override {
            auto [_, inserted] = visited.insert(&dim);
            if (inserted) DimVisitor::visit(dim.getOp()->output);
        }
        void visit(const SplitLikeOp::Input& dim) override {
            auto [_, inserted] = visited.insert(&dim);
            if (inserted) {
                DimVisitor::visit(dim.getOp()->outputLhs);
                DimVisitor::visit(dim.getOp()->outputRhs);
            }
        }
        void visit(const MergeLikeOp::Input& dim) override {
            auto [_, inserted] = visited.insert(&dim);
            if (inserted) {
                sol.addConstraint(dim.size());
                DimVisitor::visit(dim.getOp()->output);
            }
        }
        using DimVisitor::visit;
    };
    visitor v { sol };
    for (const Dimension& dim: getUnderlyingDimensions()) {
        v.visit(dim);
    }
    for (auto it: interface) {
        sol.addConstraint(it->size());
    }
    for (auto it: manipulations) {
        sol.addConstraint(it->size());
    }
    return sol.solve(Size::Product(getUnderlyingDimensions() | std::views::transform(&Dimension::size)), Size::Product(getInterfaceShape()));
}

std::size_t TensorView::getFLOPs(const ConcreteConsts& consts) const {
    std::size_t outerLoopsIterations = getInterfaceShape().totalSize().eval<std::size_t>(consts);
    std::size_t innerLoopsIterations = getManipulations().size() > 0 ? Size::Product(getManipulations() | std::views::transform([](const MapReduce *op) -> const Size& { return op->size(); })).eval<std::size_t>(consts) : 1;
    std::size_t mult = getUnderlyingTensors().size() - 1;
    bool hasDivBy = forwardAccess.divBy.has_value();
    // Multiplication + Addition + Division
    return outerLoopsIterations * innerLoopsIterations * (mult + hasDivBy) + outerLoopsIterations * (innerLoopsIterations - 1);
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
    return TensorArrayToString(tensors | std::views::transform(&PureTensor::getDimensions), ctx);
}

TensorView::TensorView(const std::vector<std::vector<Dimension>>& tensors) {
    for (std::size_t tId = 0; auto&& tensor: tensors) {
        auto name = "in_" + std::to_string(tId);
        this->tensors.emplace_back(std::move(name), tensor);
        ++tId;
    }

    Graph::Builder builder;
    builder.addTopmost(tensors | std::views::join);
    Graph graph = builder.build();
    auto& outputIterators = graph.getOutputIterators();
    auto& mapReduceIterators = graph.getMapReduceIterators();

    std::vector<Dimension> interfaceDimensions;
    std::ranges::copy(outputIterators, std::back_inserter(interfaceDimensions));

    auto forwardEval = DimensionEvaluator(graph, this->tensors);
    forwardEval.makeVars(interfaceDimensions);
    for (auto r: mapReduceIterators) {
        forwardEval.reduceAt(r);
    }
    forwardEval.adjustReductionOrder();
    forwardAccess = forwardEval.toAccess(TensorExpression::Output, interfaceDimensions);
    for (std::size_t tId = 0; auto&& tensor: this->tensors) {
        // KAS_DEBUG("Differentiating input {}...", tId);
        auto backwardEval = DimensionEvaluator(graph, this->tensors);
        backwardEval.makeVars(tensor.getDimensions());
        backwardEval.fillWithReductions();
        backwardEval.adjustReductionOrder();
        backwardAccesses.emplace_back(backwardEval.toAccess(tId, interfaceDimensions));
        ++tId;
    }

    interface = std::move(outputIterators);
    manipulations = std::move(mapReduceIterators);
}

} // namespace kas
