#pragma once

#include "KAS/Core/Size.hpp"


namespace kas {

// A simplified problem for estimating FLOPs.
// It is all about performing numel-increasing and numel-decreasing Ops.
// For sure, we should perform numel-decreasing Ops (Reduce) as early as possible, and defer numel-increasing Ops (Expand and Unfold) as late as possible.
// But the fact is, there are dependencies between them. We have to perform certain numel-increase before numel-decrease.
// This can be represented by a bipartite graph.
struct FLOPsGame {
    static constexpr std::size_t Infinity = std::numeric_limits<std::size_t>::max();

    const BindingContext& ctx;
    Size inputSize;
    std::vector<Size> increase, decrease;
    // decrease -> increase
    std::vector<std::vector<bool>> dependencies;

    struct State {
        const FLOPsGame& game;
        Size current;
        std::vector<bool> determinedIncrease, determinedDecrease;
        inline bool decreaseDetermined(std::size_t i) const { return determinedDecrease[i]; }
        std::size_t getNextDecrease(std::size_t i) const;
        State(const FLOPsGame& game);
        std::size_t FLOPs() const;
        // This means: we have performed some reductions. Now perform some trials to find the least remaining FLOPs.
        std::size_t remainingFLOPs(const State& base) const;
        // This means: based on base, we want to perform more reductions in a single round.
        // And we are only allowed to try from decrease[nowDecreasing].
        std::size_t experimentRemainingFLOPs(const State& base, std::size_t nowDecreasing) const;
    };

    std::size_t FLOPs() const;
};

} // namespace kas
