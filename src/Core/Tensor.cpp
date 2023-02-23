#include <iterator>
#include <limits>
#include <map>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Common.hpp"

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
    struct DimensionEvaluator {
        std::map<Dimension, IteratorValue> memoize;
        std::map<const Iterator *, IteratorValue> outer;
        std::map<const MapReduceOp *, IteratorValue> inner;
        IteratorValue dfs(Dimension dim) {
            if (auto it = memoize.find(dim); it != memoize.end()) {
                return it->second;
            }
            IteratorValue result;
            if (auto it = dynamic_cast<const RepeatLikePrimitiveOp *>(dim.get()); it) {
                auto out = dfs(it->output);
                result = it->value(out);
            } else if (auto it = dynamic_cast<const SplitLikePrimitiveOp *>(dim.get()); it) {
                auto outLeft = dfs(it->outputLhs);
                auto outRight = dfs(it->outputRhs);
                result = it->value(outLeft, outRight);
            } else if (auto it = dynamic_cast<const MergeLikePrimitiveOp *>(dim.get()); it) {
                auto out = dfs(it->output);
                result = it->value(out, it->order);
            } else if (auto it = dynamic_cast<const Iterator *>(dim.get()); it) {
                // Here we have not figured out the order of the iterators. We have to wait until all the iterators are collected.
                result = VariableValueNode::Create(std::numeric_limits<std::size_t>::max(), it->getName());
                auto [ptr, inserted] = outer.insert({it, result});
                if (!inserted) {
                    result = ptr->second;
                }
            } else if (auto it = dynamic_cast<const MapReduceOp *>(dim.get()); it) {
                // Same reason.
                result = VariableValueNode::Create(std::numeric_limits<std::size_t>::max(), it->getName());
                auto [ptr, inserted] = inner.insert({it, result});
                if (!inserted) {
                    result = ptr->second;
                }
            } else {
                KAS_CRITICAL("When evaluating access, encountered unknown dimension type: {}", typeid(*dim.get()).name());
            }
            memoize.insert({dim, result});
            return result;
        }
        void fill(std::vector<const Iterator *> interface, std::vector<const MapReduceOp *> manipulations) {
            std::ranges::copy(outer | std::views::transform([](auto&& pair) { return pair.first; }), std::back_inserter(interface));
            std::ranges::copy(inner | std::views::transform([](auto&& pair) { return pair.first; }), std::back_inserter(manipulations));
            std::ranges::sort(interface, [](const Iterator *lhs, const Iterator *rhs) {
                return lhs->getIndex() < rhs->getIndex();
            });
            std::ranges::sort(manipulations, [](const MapReduceOp *lhs, const MapReduceOp *rhs) {
                return lhs->getPriority() < rhs->getPriority();
            });
            std::size_t index = 0;
            for (auto&& it: interface)
                outer[it].as<VariableValueNode>().variableId = index++;
            for (auto&& it: manipulations)
                inner[it].as<VariableValueNode>().variableId = index++;
        }
    };
}

TensorView::TensorView(const std::vector<std::vector<Dimension>>& tensors) {
    auto eval = DimensionEvaluator();
    for (std::size_t tId = 0; auto&& tensor: tensors) {
        auto access = std::vector<IteratorValue>();
        std::ranges::copy(tensor | std::views::transform([&](const auto& dim) {
            return eval.dfs(dim);
        }), std::back_inserter(access));
        auto name = "in_" + std::to_string(tId);
        this->tensors.emplace_back(std::move(name), tensor, std::move(access));
        ++tId;
    }
    eval.fill(interface, manipulations);
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
        IndentSpaces(ss, depth);
        fmt::format_to(SSIt(ss), "for (int {0} = 0; {0} < {1}; {0}++) {{\n", it->getName(), it->size().toString(ctx));
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
