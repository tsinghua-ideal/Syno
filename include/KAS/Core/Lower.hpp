#pragma once

#include <set>
#include <variant>

#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class LocalityOptimizer {
    const Graph& graph;
public:
    LocalityOptimizer(const Graph& graph): graph { graph } {}
    void permuteWeightDimensions(std::vector<std::vector<Dimension>>& tensors) const;
};

class TensorExpressionDifferentiator final: public ValuedTensorExpressionVisitor<
    TensorExpressionDifferentiator,
    TensorExpression
> {
    TensorExpression::Position position;
public:
    TensorExpressionDifferentiator(TensorExpression::Position position): position { position } {
        KAS_ASSERT(position != TensorExpression::Output);
    }
    TensorExpression visits(IntegerTensorExpression& expr);
    TensorExpression visits(TensorTensorExpression& expr);
    TensorExpression visits(BinaryOpTensorExpression& expr);
    TensorExpression differentiate(TensorExpression& expr);
};

class DimensionEvaluator {
    const Graph& graph;
    const std::vector<PureTensor>& inputTensors;

    std::map<Dimension, Valuation, Dimension::AddressLessThan> values;
    std::set<Dimension, Dimension::AddressLessThan> unknownDimensions;
    std::set<Dimension, Dimension::AddressLessThan> freeCandidates;

    std::vector<IteratorValue> outerLoops;
    std::vector<IteratorValue> innerLoops;
    std::vector<Size> innerLoopsShape;

    // Description of the expression.
    TensorExpression expression;
    std::optional<Size> divBy;

    // Extract the values of branches of this Op.
    template<Vertex V>
    decltype(auto) getBranches(const V& v) {
        using Values = typename V::OpType::Values;
        return Values::FillBy([&](std::size_t branch) {
            // We have to make sure branches are in order!
            return values[v[static_cast<V::BranchType>(branch)]];
        });
    }

    // Update the values.
    template<Vertex V>
    void setBranches(const V& v, const typename V::OpType::Values& newValues) {
        for (std::uint8_t branch = 0; branch < V::OpType::BranchCount; ++branch) {
            Dimension d = v[static_cast<V::BranchType>(branch)];
            values[d] = newValues[static_cast<V::BranchType>(branch)];
        }
    }

    // Propagate values.
    struct Propagator {
        DimensionEvaluator& eval;
        Propagator(DimensionEvaluator& eval): eval { eval } {}
        Propagator(const Propagator&) = delete;
        Propagator(Propagator&&) = delete;
        template<Vertex V>
        void operator()(const V& v, auto from) {
            auto knownValues = eval.getBranches(v);
            auto newValues = v.op.value(knownValues);
            constexpr std::size_t cnt = V::OpType::BranchCount;
            for (std::uint8_t branch = 0; branch < cnt; ++branch) {
                auto& knownValue = knownValues[branch];
                auto& newValue = newValues[branch];
                newValue.assertCanBeConvertedFrom(knownValue);
                Dimension which = v[static_cast<V::BranchType>(branch)];
                if (newValue.isRefined(knownValue)) {
                    if (newValue.type() == Valuation::Type::Valued) {
                        // We have assigned this Dimension.
                        eval.unknownDimensions.erase(which);
                    }
                    if (knownValue.type() == Valuation::Type::Unoriented) {
                        // This Dimension can no longer be free.
                        eval.freeCandidates.erase(which);
                    }
                }
                eval.values[which] = newValue;
            }
            for (std::uint8_t branch = 0; branch < cnt; ++branch) {
                auto& knownValue = knownValues[branch];
                auto& newValue = newValues[branch];
                if (newValue.isRefined(knownValue)) {
                    // DFS.
                    v.visitAdjacent(static_cast<V::BranchType>(branch)).match(*this, *this, *this);
                }
            }
        }
    };

    // Assign a dimension with an iterator value, and propagate this value through the graph.
    void assign(Dimension dim, IteratorValue value);
    // Obtain the iterator values for specified dimensions.
    std::vector<IteratorValue> extractValues(const std::vector<Dimension>& dims) const;
public:
    DimensionEvaluator(const Graph& graph, const std::vector<PureTensor>& inputTensors, TensorExpression blending);
    // Assign a dimension with an outer loop iterator.
    void makeVar(Dimension dim);
    // Do the above for all specified dimensions in the specified order.
    void makeVars(const std::vector<Dimension>& dims);
    // Assign a dimension with an inner loop iterator.
    void reduceAt(Dimension dim);
    KAS_STATISTICS_DEF(
        CannotFindInputDimensionToReduce
    )
    // Actually do the differentiation, by evaluating the best inner loops.
    void fillWithReductions();
    // Adjust reduction order to speed up. We need to make sure all dimensions are assigned before this.
    void adjustReductionOrder();
    // Convert this evaluator to an AbstractAccess.
    AbstractAccess toAccess(int position, const std::vector<Dimension>& output);
};

} // namespace kas
