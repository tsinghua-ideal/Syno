#pragma once

#include <queue>
#include <set>
#include <variant>

#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

namespace Derivative {

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
    void visit(const Iterator& dim) override;
    void visit(const MapReduceOp& dim) override;
    void visit(const RepeatLikeOp::Input& dim) override;
    void visit(const SplitLikeOp::Input& dim) override;
    void visit(const MergeLikeOp::Input& dim) override;
    void visit(const Dimension& dim);

public:
    void add(const Dimension& dim);
    AdditionalMetadata build();
};

class DimensionEvaluator {
    const std::map<Dimension, OpAbove, Dimension::AddressLessThan>& theOtherEndOfEdge;
    std::map<Dimension, IteratorValue, Dimension::AddressLessThan> values;
    std::vector<IteratorValue> outerLoops;
    std::vector<IteratorValue> innerLoops;
    std::vector<Size> innerLoopsShape;
    std::set<Dimension, Dimension::AddressLessThan> remaining;

    std::queue<std::pair<Dimension, Direction>> bfsQueue; // false is down, true is up.
    std::map<Dimension, int, Dimension::AddressLessThan> bfsVisited;
    int highestPriority = std::numeric_limits<int>::min();
    std::optional<Dimension> bestCandidate;

    // This visitor does the actual traversal of the graph.
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
                        int newPriority = priority + currentPriority;
                        eval.highestPriority = std::max(eval.highestPriority, newPriority);
                        if (newPriority == eval.highestPriority) { // Only if the next dimension is superior, add it.
                            auto [it, inserted] = eval.bfsVisited.emplace(dim, newPriority);
                            if (!inserted) { // Maybe we have encountered a cycle. But as long as it has the same priority, there is no cycle.
                                if (it->second != newPriority) {
                                    KAS_CRITICAL("We should not have visited the dimension before. Check for loops.");
                                }
                            } else { // We can now add it to queue.
                                eval.bfsQueue.emplace(dim, dir);
                                eval.bestCandidate = dim;
                            }
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
        inline Visitor(DimensionEvaluator& eval, const Dimension& current):
            eval { eval },
            current { current }
        {}
    };

    // This visitor walks down along a dimension, which is an input dimension of an Op. The visitor visits this Op.
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
            Dimension inputLhs = op->getInputL(), inputRhs = op->getInputR();
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
    // Construct a WalkDownVisitor and visit the dimension.
    template<bool isAssigning>
    void walkDown(const Dimension& dim) {
        WalkDownVisitor<isAssigning>(*this, dim).visit(dim);
    }

    // This visitor walks up along a dimension, which is an output dimension of an Op. The visitor visits this Op.
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
            Dimension inputLhs = op->getInputL(), inputRhs = op->getInputR();
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
    // Construct a WalkUpVisitor and visit the dimension. This uses the OpAbove's collected earlier.
    template<bool isAssigning>
    void walkUp(const Dimension& dim) {
        theOtherEndOfEdge.at(dim).visit(WalkUpVisitor<isAssigning>(*this, dim));
    }

    // Assign a dimension with an iterator value, and propagate this value through the graph.
    void assign(const Dimension& dim, IteratorValue value);
public:
    DimensionEvaluator(const std::map<Dimension, OpAbove, Dimension::AddressLessThan>& theOtherEndOfEdge);
    // Assign a dimension with an outer loop iterator.
    void makeVar(const Dimension& dim);
    // Do the above for all specified dimensions in the specified order.
    void makeVars(const std::vector<Dimension>& dims);
    // Assign a dimension with an inner loop iterator.
    void reduceAt(const Dimension& dim);
    // Actually do the differentiation, by evaluating the best inner loops.
    void fillWithReductions();
    // Obtain the iterator values for specified dimensions.
    std::vector<IteratorValue> extractValues(const std::vector<Dimension>& dims) const;
    // Convert this evaluator to an AbstractAccess.
    AbstractAccess toAccess(int position, const std::vector<PureTensor>& inputs, const std::vector<Dimension>& output);
};

} // namespace Diff

} // namespace kas
