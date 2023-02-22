#include <iterator>
#include <map>
#include <ranges>
#include <sstream>
#include <string>

#include <fmt/core.h>
#include <vector>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Tensor.hpp"

namespace kas {

std::string PureTensor::shapeToString(const BindingContext& ctx) const {
    return getShape().toString(ctx);
}

std::string PureTensor::accessToString(const BindingContext& ctx) const {
    auto printer = IteratorValuePrinter(ctx);
    return VectorToString(access | std::views::transform([&](const IteratorValue& value) {
        return printer.toString(value);
    }));
}

namespace {
    auto SSIt(std::stringstream& ss) -> decltype(auto) {
        return std::ostreambuf_iterator<char>(ss);
    }
    void IndentSpaces(std::stringstream& ss, std::size_t indent) {
        fmt::format_to(SSIt(ss), "{:{}}", "", 4 * indent);
    }
}

std::string TensorView::printInnerLoops(const BindingContext& ctx, std::size_t indent, std::string_view outputName) const {
    std::stringstream ss;
    const auto recursion = [this, &ctx, &ss](const auto& self, std::size_t manipId, std::size_t depth) -> std::string {
        if (manipId-- == 0) {
            auto r = tensors | std::views::transform([&](const auto& tensor) {
                return tensor.getName() + tensor.accessToString(ctx);
            });
            // TODO: other blending schemes other than multiplication
            return fmt::format("{}", fmt::join(r, " * "));
            std::stringstream innerSs;
        }

        auto& m = manipulations[manipId];
        std::string tempName = "temp_" + m->getName();

        // float temp_ri_idx = 0;
        IndentSpaces(ss, depth);
        fmt::format_to(SSIt(ss), "float {} = 0;\n", tempName);

        // for (int ri_idx = 0; ri_idx < size_ri_idx; ri_idx++) {
        IndentSpaces(ss, depth);
        fmt::format_to(SSIt(ss), "for (int {0} = 0; {0} < {1}; {0}++) {{\n", m->getName(), m->size().toString(ctx));

        // Generate inner loops, and obtain the temporary variable.
        std::string inner = self(self, manipId, depth + 1);

        IndentSpaces(ss, depth + 1);
        // Can be other reduce than sum. TODO.
        fmt::format_to(SSIt(ss), "{} += {}({});\n", tempName, m->whatMap(), inner);

        IndentSpaces(ss, depth);
        ss << "}\n";

        return tempName;
    };
    std::string lastTemp = recursion(recursion, manipulations.size(), indent);
    IndentSpaces(ss, indent);
    fmt::format_to(SSIt(ss), "{}{} = {};\n", outputName, interfaceAccessToString(ctx), lastTemp);
    return ss.str();
}

namespace {
    struct DimensionEvaluator: public IteratorValueVisitor {
        std::map<Dimension, IteratorValue> memoize;
        std::vector<const Iterator *> outer;
        std::vector<const MapReduceOp *> inner;
        void visit(VariableValueNode& value) {

        }
        void visit(ConstValueNode& value) {

        }
        void visit(ImmediateValueNode& value) {

        }
        void visit(BinaryOpValueNode& value) {

        }
        void visit(IntervalBoundValueNode& value) {

        }
        IteratorValue dfs(Dimension dim) {
            // TODO
        }
    };
}

TensorView::TensorView(const std::vector<std::vector<Dimension>>& tensors)
{
    auto eval = DimensionEvaluator();
    for (auto&& tensor: tensors) {

    }
    interface = std::move(eval.outer);
    manipulations = std::move(eval.inner);
}

std::string TensorView::shapeToString(const BindingContext& ctx) const {
    auto s1 = IteratorShapeView(interface).toString(ctx);
    if (!manipulations.empty()) {
        auto s2 = ReduceIteratorShapeView(manipulations).toString(ctx);
        return s1 + " with reduced " + s2;
    }
    return s1;
}

std::string TensorView::interfaceAccessToString(const BindingContext& ctx) const {
    return VectorToString(interface
        | std::views::transform([&](const Iterator *it) {
            return it->getName();
        })
    );
}

std::string TensorView::actualAccessToString(const BindingContext& ctx) const {
    std::stringstream ss;
    // Here the outer loops are exactly the interface iterators.
    ss << VectorToString(tensors
        | std::views::transform([](const PureTensor& t) -> ShapeView {
            return t.getShape();
        })
        | std::views::join
        | std::views::transform([&](const Size& size) {
            return size.toString(ctx);
        })
    );
    for (const auto& m: manipulations) {
        ss << " with " << m->whatMap() << " mapped";
        ss << " with " << m->getName() << " " << m->whatReduce() << " reduced";
    }
    return ss.str();
}

std::string TensorView::printNestedLoops(const BindingContext& ctx, std::string_view outputName) const {
    std::stringstream ss;
    std::size_t depth = 0;

    for (auto it: interface) {
        auto name = "i_" + std::to_string(it->getIndex());
        IndentSpaces(ss, depth);
        fmt::format_to(SSIt(ss), "for (int {0} = 0; {0} < {1}; {0}++) {{\n", name, it->size().toString(ctx));
        ++depth;
    }

    ss << printInnerLoops(ctx, depth, outputName);

    while (depth --> 0) {
        IndentSpaces(ss, depth);
        ss << "}\n";
    }
    return ss.str();
}

} // namespace kas
