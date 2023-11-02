#pragma once

#include "KAS/Core/FLOPsGame.hpp"
#include "KAS/Core/Graph.hpp"


namespace kas {

// FLOPsGame that includes ShareOp's.
class ExtendedFLOPsGame {
    struct Adjacency {
        std::vector<std::size_t> increaseIndices;
        std::vector<std::size_t> decreaseIndices;
    };
    const BindingContext& ctx;
    Size inputSize;
    std::vector<Size> increase, decrease;
    std::vector<std::vector<bool>> dependencies;
    // Key: Share RHS, Iterator (which is also a special Expand). Basically what appears in weights.
    // Value: Adjacency. The indices of Unfolds and Expands that must be done before contraction, and the indices of reductions that must be done after contraction.
    Graph::DimensionMap<Adjacency> sharedDependencies;
public:
    ExtendedFLOPsGame(const BindingContext& ctx, Size inputSize, const Graph& graph);
    // Look up adjacencies, and augment the dependencies. That is, during a contraction, all the dims in a single weight are added, so one decrease can depend on more increase's then explicitly derived.
    FLOPsGame getGameWithWeights(const std::vector<std::vector<Dimension>>& weights) const;
};

} // namespace kas
