#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Common.hpp"

namespace kas {

std::string PureTensor::shapeToString(const BindingContext& ctx) const {
    return getShape().toString(ctx);
}

std::string AbstractAccess::outerLoopsIteratorsToString() const {
    return VectorToString(outerLoops
        | std::views::transform([](const IteratorValue& it) {
            return it.as<VariableValueNode>().name;
        })
    );
}

std::string AbstractAccess::innerLoopsIteratorsToString() const {
    return VectorToString(innerLoops
        | std::views::transform([](const IteratorValue& it) {
            return it.as<VariableValueNode>().name;
        })
    );
}

std::string AbstractAccess::accessToString(const BindingContext& ctx, int pos) const {
    return VectorToString((pos == Output ? output : inputs.at(pos))
        | std::views::transform([&](const IteratorValue& it) {
            return it.toString(ctx);
        })
    );
}

std::string AbstractAccess::statementToString(const BindingContext& ctx) const {
    return fmt::format("{}", fmt::join(
        std::views::iota(std::size_t{0}, inputs.size())
        | std::views::transform([&](std::size_t pos) {
            if (pos == position) {
                return fmt::format("grad_out{}", accessToString(ctx, Output));
            } else {
                return fmt::format("in_{}{}", pos, accessToString(ctx, pos));
            }
        }),
    " * "));
}

std::string AbstractAccess::targetEntryToString() const {
    if (position == Output) {
        return fmt::format("out{}", outerLoopsIteratorsToString());
    } else {
        return fmt::format("grad_in_{}{}", position, outerLoopsIteratorsToString());
    }
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

    auto& access = pos == AbstractAccess::Output ? forwardAccess : backwardAccesses.at(pos);

    for (auto it: access.outerLoops) {
        const auto& var = it.as<VariableValueNode>();
        IndentSpaces(ss, depth);
        fmt::format_to(SSIt(ss),
            "for (int {0} = 0; {0} < {1}; {0}++) {{\n",
            var.name,
            (pos == AbstractAccess::Output ? interface.at(depth)->size() : tensors.at(pos).getShape()[depth]).toString(ctx)
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

namespace {
    struct OpAbove {
        std::variant<std::monostate, const RepeatLikeOp *, std::pair<const SplitLikeOp *, Order>, const MergeLikeOp *> op;
        template<typename F>
        void visit(F&& f) const {
            std::visit(std::forward<F>(f), op);
        }
    };
    // In the original graph, each Dimension serves as an edge. It provides easy access for the Op below it (which has the Dimension as input), but cannot access the Op above it. This struct stores the Op above each Dimension.
    // And the output/reduce iterators as well.
    struct AdditionalMetadata {
        std::map<Dimension, OpAbove, Dimension::AddressLessThan> theOtherEndOfEdge;
        std::vector<const Iterator *> outputIterators;
        std::vector<const MapReduceOp *> mapReduceIterators;
    };

    // DFS the graph to build up the relations.
    class DimensionGraphBuilder final: public DimVisitor {
        AdditionalMetadata extra;

        OpAbove parent;
        void visit(const Iterator& dim) override {
            extra.outputIterators.push_back(&dim);
        }
        void visit(const MapReduceOp& dim) override {
            extra.mapReduceIterators.push_back(&dim);
        }
        void visit(const RepeatLikeOp::Input& dim) override {
            auto op = dim.getOp();
            parent = { op };
            visit(op->output);
        }
        void visit(const SplitLikeOp::Input& dim) override {
            auto op = dim.getOp();
            parent = { std::make_pair(op, Order::Left) };
            visit(op->outputLhs);
            parent = { std::make_pair(op, Order::Right) };
            visit(op->outputRhs);
        }
        void visit(const MergeLikeOp::Input& dim) override {
            auto op = dim.getOp();
            parent = { op };
            visit(op->output);
        }
        void visit(const Dimension& dim) {
            auto [it, inserted] = extra.theOtherEndOfEdge.insert({dim, parent});
            if (inserted) {
                DimVisitor::visit(dim);
            }
        }

    public:
        void add(const Dimension& dim) {
            parent = { std::monostate{} };
            visit(dim);
        }
        AdditionalMetadata build() {
            std::ranges::sort(extra.outputIterators, [](const Iterator *lhs, const Iterator *rhs) {
                return lhs->getIndex() < rhs->getIndex();
            });
            std::ranges::sort(extra.mapReduceIterators, [](const MapReduceOp *lhs, const MapReduceOp *rhs) {
                return lhs->getPriority() < rhs->getPriority();
            });
            return std::move(extra);
        }
    };

    class DimensionEvaluator {
        const std::map<Dimension, OpAbove, Dimension::AddressLessThan>& theOtherEndOfEdge;
        std::map<Dimension, IteratorValue, Dimension::AddressLessThan> values;
        std::vector<IteratorValue> outerLoops;
        std::vector<IteratorValue> innerLoops;
        std::vector<Size> innerLoopsShape;
        std::set<Dimension, Dimension::AddressLessThan> remaining;

        enum class Direction: bool {
            Down = false,
            Up = true,
        };
        std::queue<std::pair<Dimension, Direction>> bfsQueue; // false is down, true is up.
        std::map<Dimension, int, Dimension::AddressLessThan> bfsVisited;
        int highestPriority = std::numeric_limits<int>::min();
        std::optional<Dimension> bestCandidate;

        class Visitor {
        protected:
            using RepeatIV = RepeatLikeOp::IteratorValues;
            using RepeatOV = RepeatLikeOp::OrderingValues;
            using SplitIV = SplitLikeOp::IteratorValues;
            using SplitOV = SplitLikeOp::OrderingValues;
            using MergeIV = MergeLikeOp::IteratorValues;
            using MergeOV = MergeLikeOp::OrderingValues;

            DimensionEvaluator& eval;
            const Dimension& current;

            template<typename... Args>
            void checkAssign(const auto *op, const auto& knownValues, Args&&... args) {
                if (knownValues.known()) return; // Evaluated.
                // We are walking down from current, so of course it has value.
                KAS_ASSERT(eval.values[current].hasValue());
                auto newValues = op->value(knownValues);
                auto checker = [&](Direction dir, const Dimension& dim, auto member) {
                    if (newValues.*member) {
                        eval.values[dim] = newValues.*member;
                        eval.remaining.erase(dim);
                        if (dir == Direction::Down) {
                            eval.walkDown<true>(dim);
                        } else {
                            eval.walkUp<true>(dim);
                        }
                    }
                };
                (checker(std::get<0>(args), std::get<1>(args), std::get<2>(args)), ...);
            }

            template<typename... Args>
            void updatePriorities(const auto *op, const auto& knownValues, auto self, Args&&... args) {
                if (knownValues.known()) return; // Evaluated.
                // We are walking down from current, so of course it has value.
                int currentPriority = eval.bfsVisited.at(current);
                auto relation = op->ordering(knownValues);
                if (relation.*self != -1) { // Make sure self is in relation.
                    auto checker = [&](Direction dir, const Dimension& dim, auto member) {
                        if (int priority = relation.*member - relation.*self; priority >= 0) { // Observe that only unevaluated dimensions will satisfy this condition.
                            eval.bfsQueue.emplace(dim, dir);
                            int newPriority = priority + currentPriority;
                            auto [_, inserted] = eval.bfsVisited.emplace(dim, newPriority);
                            KAS_ASSERT(inserted, "We should not have visited the dimension before. Check for loops.");
                            eval.highestPriority = std::max(eval.highestPriority, newPriority);
                            if (newPriority == eval.highestPriority) {
                                eval.bestCandidate = dim;
                            }
                        }
                    };
                    (checker(std::get<0>(args), std::get<1>(args), std::get<3>(args)), ...);
                }
            }

            template<bool isAssigning, typename... Args>
            void walk(const auto *op, const auto& knownValues, auto self, Args&&... args) {
                if constexpr (isAssigning) {
                    checkAssign(op, knownValues, std::forward<Args>(args)...);
                } else {
                    updatePriorities(op, knownValues, self, std::forward<Args>(args)...);
                }
            }

        public:
            Visitor(DimensionEvaluator& eval, const Dimension& current):
                eval { eval },
                current { current }
            {}
        };

        template<bool isAssigning>
        class WalkDownVisitor: public DimVisitor, public Visitor {
            void visit(const RepeatLikeOp::Input& dim) override {
                auto op = dim.getOp();
                auto knownValues = RepeatLikeOp::IteratorValues {
                    eval.values[current],
                    eval.values[op->output],
                };
                walk<isAssigning>(op, knownValues, &RepeatOV::input,
                    std::tuple{Direction::Down, op->output, &RepeatIV::output, &RepeatOV::output}
                );
            }
            void visit(const SplitLikeOp::Input& dim) override {
                auto op = dim.getOp();
                auto knownValues = SplitLikeOp::IteratorValues {
                    eval.values[current],
                    eval.values[op->outputLhs],
                    eval.values[op->outputRhs],
                };
                walk<isAssigning>(op, knownValues, &SplitOV::input,
                    std::tuple{Direction::Down, op->outputLhs, &SplitIV::outputLhs, &SplitOV::outputLhs},
                    std::tuple{Direction::Down, op->outputRhs, &SplitIV::outputRhs, &SplitOV::outputRhs}
                );
            }
            void visit(const MergeLikeOp::Input& dim) override {
                auto op = dim.getOp();
                auto [inputLhs, inputRhs] = op->getInputs();
                auto order = dim.getOrder();
                auto knownValues = MergeLikeOp::IteratorValues {
                    eval.values[inputLhs],
                    eval.values[inputRhs],
                    eval.values[op->output],
                };
                walk<isAssigning>(op, knownValues,
                    order == Order::Left ? // Determine which input is self.
                        &MergeOV::inputLhs : &MergeOV::inputRhs,
                    std::tuple{Direction::Down, op->output, &MergeIV::output, &MergeOV::output},
                    order == Order::Left ? // Propagate to the other side.
                        std::tuple{Direction::Up, inputRhs, &MergeIV::inputRhs, &MergeOV::inputRhs} :
                        std::tuple{Direction::Up, inputLhs, &MergeIV::inputLhs, &MergeOV::inputLhs}
                );
            }
        public:
            using Visitor::Visitor;
            using DimVisitor::visit;
        };
        template<bool isAssigning>
        void walkDown(const Dimension& dim) {
            WalkDownVisitor<isAssigning>(*this, dim).visit(dim);
        }

        template<bool isAssigning>
        class WalkUpVisitor: public Visitor {
        public:
            void operator()(std::monostate) {}
            void operator()(const RepeatLikeOp *op) {
                auto knownValues = RepeatLikeOp::IteratorValues {
                    eval.values[op->getInput()],
                    eval.values[current],
                };
                walk<isAssigning>(op, knownValues, &RepeatOV::output,
                    std::tuple{Direction::Up, op->getInput(), &RepeatIV::input, &RepeatOV::input}
                );
            }
            void operator()(std::pair<const SplitLikeOp *, Order> opAndOrder) {
                auto& [op, order] = opAndOrder;
                auto knownValues = SplitLikeOp::IteratorValues {
                    eval.values[op->getInput()],
                    eval.values[op->outputLhs],
                    eval.values[op->outputRhs],
                };
                walk<isAssigning>(op, knownValues,
                    order == Order::Left ? // Determine side of self.
                        &SplitOV::outputLhs : &SplitOV::outputRhs,
                    std::tuple{Direction::Up, op->getInput(), &SplitIV::input, &SplitOV::input},
                    order == Order::Left ? // Propagate to the other side.
                        std::tuple{Direction::Down, op->outputRhs, &SplitIV::outputRhs, &SplitOV::outputRhs} :
                        std::tuple{Direction::Down, op->outputLhs, &SplitIV::outputLhs, &SplitOV::outputLhs}
                );
            }
            void operator()(const MergeLikeOp *op) {
                auto [inputLhs, inputRhs] = op->getInputs();
                auto knownValues = MergeLikeOp::IteratorValues {
                    eval.values[inputLhs],
                    eval.values[inputRhs],
                    eval.values[current],
                };
                walk<isAssigning>(op, knownValues, &MergeOV::output,
                    std::tuple{Direction::Up, inputLhs, &MergeIV::inputLhs, &MergeOV::inputLhs},
                    std::tuple{Direction::Up, inputRhs, &MergeIV::inputRhs, &MergeOV::inputRhs}
                );
            }
            using Visitor::Visitor;
        };
        template<bool isAssigning>
        void walkUp(const Dimension& dim) {
            theOtherEndOfEdge.at(dim).visit(WalkUpVisitor<isAssigning>(*this, dim));
        }

        void assign(const Dimension& dim, IteratorValue value) {
            auto [it, _] = values.insert_or_assign(dim, IteratorValue{});
            KAS_ASSERT(!it->second);
            it->second = value;
            remaining.erase(dim);
            walkDown<true>(dim);
            walkUp<true>(dim);
        }
    public:
        DimensionEvaluator(const std::map<Dimension, OpAbove, Dimension::AddressLessThan>& theOtherEndOfEdge):
            theOtherEndOfEdge { theOtherEndOfEdge }
        {
            // Initialize the remaining set.
            std::ranges::copy(theOtherEndOfEdge | std::views::transform([](auto&& kv) { return kv.first; }), std::inserter(remaining, remaining.end()));
        }
        void makeVar(const Dimension& dim) {
            outerLoops.emplace_back(VariableValueNode::Create(false, outerLoops.size(), "i_" + std::to_string(outerLoops.size())));
            assign(dim, outerLoops.back());
        }
        void makeVars(const std::vector<Dimension>& dims) {
            for (const auto& dim: dims) {
                makeVar(dim);
            }
        }
        void reduceAt(const Dimension& dim) {
            innerLoopsShape.emplace_back(dim.size());
            innerLoops.emplace_back(VariableValueNode::Create(true, innerLoops.size(), "ri_" + std::to_string(innerLoops.size())));
            assign(dim, innerLoops.back());
        }
        void fillWithReductions() {
            while (!remaining.empty()) {
                Dimension remainder = *remaining.begin();
                KAS_ASSERT(!values[remainder]);
                bfsQueue.emplace(remainder, Direction::Down);
                bfsQueue.emplace(remainder, Direction::Up);
                bfsVisited.emplace(remainder, 0);
                highestPriority = 0;
                bestCandidate = remainder;
                while (!bfsQueue.empty()) {
                    auto [dim, direction] = bfsQueue.front();
                    bfsQueue.pop();
                    if (bfsVisited.at(dim) < highestPriority) continue;
                    switch (direction) {
                    case Direction::Down:
                        walkDown<false>(dim);
                        break;
                    case Direction::Up:
                        walkUp<false>(dim);
                        break;
                    }
                }
                reduceAt(*bestCandidate);

                // Clean up for next iteration.
                bfsQueue = decltype(bfsQueue) {};
                bfsVisited.clear();
                highestPriority = std::numeric_limits<int>::min();
                bestCandidate = std::nullopt;
            }
        }
        std::vector<IteratorValue> extractValues(const std::vector<Dimension>& dims) const {
            KAS_ASSERT(remaining.empty());
            std::vector<IteratorValue> result;
            std::ranges::move(dims | std::views::transform([&](const auto& dim) { return values.at(dim); }), std::back_inserter(result));
            return result;
        }
        AbstractAccess toAccess(int position, const std::vector<PureTensor>& inputs, const std::vector<Dimension>& output) {
            std::vector<std::vector<IteratorValue>> inputsAccesses;
            std::ranges::move(inputs | std::views::transform([&](const auto& tensor) { return extractValues(tensor.getDimensions()); }), std::back_inserter(inputsAccesses));
            return AbstractAccess {
                .position = position,
                .outerLoops = std::move(outerLoops),
                .innerLoops = std::move(innerLoops),
                .innerLoopsShape = std::move(innerLoopsShape),
                .inputs = std::move(inputsAccesses),
                .output = extractValues(output),
            };
        }
    };
}

TensorView::TensorView(const std::vector<std::vector<Dimension>>& tensors) {
    for (std::size_t tId = 0; auto&& tensor: tensors) {
        auto name = "in_" + std::to_string(tId);
        this->tensors.emplace_back(std::move(name), tensor);
        ++tId;
    }

    DimensionGraphBuilder builder;
    for (auto&& dim: tensors | std::views::join) {
        builder.add(dim);
    }
    auto [theOtherEndOfEdge, outputIterators, mapReduceIterators] = builder.build();
    interface = std::move(outputIterators);
    manipulations = std::move(mapReduceIterators);

    std::vector<Dimension> interfaceDimensions;
    std::ranges::copy(interface | std::views::transform([](const Iterator *it) { return it; }), std::back_inserter(interfaceDimensions));

    auto forwardEval = DimensionEvaluator(theOtherEndOfEdge);
    forwardEval.makeVars(interfaceDimensions);
    for (auto r: manipulations) {
        forwardEval.reduceAt(r);
    }
    forwardAccess = forwardEval.toAccess(AbstractAccess::Output, this->tensors, interfaceDimensions);
    for (std::size_t tId = 0; auto&& tensor: this->tensors) {
        auto backwardEval = DimensionEvaluator(theOtherEndOfEdge);
        backwardEval.makeVars(tensor.getDimensions());
        backwardEval.fillWithReductions();
        backwardAccesses.emplace_back(backwardEval.toAccess(tId, this->tensors, interfaceDimensions));
        ++tId;
    }
}

} // namespace kas
