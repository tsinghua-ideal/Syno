#pragma once

#include <set>
#include <variant>

#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

class DimensionEvaluator {
    const Graph& graph;

    std::map<Dimension, Valuation, Dimension::AddressLessThan> values;
    std::set<Dimension, Dimension::AddressLessThan> unknownDimensions;
    std::set<Dimension, Dimension::AddressLessThan> freeCandidates;

    std::vector<IteratorValue> outerLoops;
    std::vector<IteratorValue> innerLoops;
    std::vector<Size> innerLoopsShape;

    // Extract the values of branches of this Op.
    template<Vertex V>
    decltype(auto) getBranches(const V& v) const {
        using Values = typename V::OpType::Values;
        return Values::FillBy([&](std::size_t branch) {
            // We have to make sure branches are in order!
            return values[v[branch]];
        });
    }

    // Update the values.
    template<Vertex V>
    void setBranches(const V& v, const typename V::OpType::Values& newValues) {
        for (std::uint8_t branch = 0; branch < V::OpType::BranchCount; ++branch) {
            Dimension d = v[branch];
            values[d] = newValues[branch];
        }
    }

    // Propagate values.
    struct Propagator {
        const DimensionEvaluator& eval;
        Propagator(const DimensionEvaluator& eval): eval { eval } {}
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
                if (newValue.isRefined(knownValue)) {
                    if (newValue.type() == Valuation::Valued) {
                        // We have assigned this Dimension.
                        eval.unknownDimensions.erase(v[branch]);
                    }
                    if (knownValue.type() == Valuation::Unoriented) {
                        // This Dimension can no longer be free.
                        eval.freeCandidates.erase(v[branch]);
                    }
                }
            }
            for (std::uint8_t branch = 0; branch < cnt; ++branch) {
                auto& knownValue = knownValues[branch];
                auto& newValue = newValues[branch];
                if (newValue.isRefined(knownValue)) {
                    // DFS.
                    v.visitAdjacent(branch).match(*this, *this, *this);
                }
            }
        }
    };

    // Assign a dimension with an iterator value, and propagate this value through the graph.
    void assign(const Dimension& dim, IteratorValue value);
public:
    DimensionEvaluator(const Graph& graph);
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

} // namespace kas
