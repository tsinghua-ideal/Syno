#pragma once

#include "KAS/Core/DimVisitor.hpp"


namespace kas {

struct ReshapeBlockNeighbors {
    using Self = ReshapeBlockNeighbors;
    using Side = std::variant<std::monostate, const MergeOp *, const Reduce *>; // nullptr is not allowed.
    Side left;
    Side right;
    struct Multiple {
        std::set<Side> lefts, rights;
        void add(const ReshapeBlockNeighbors& neighbors);
        template<std::ranges::input_range R>
        requires std::same_as<std::ranges::range_value_t<R>, Self>
        void add(R&& neighbors) {
            for (const auto& neighbor: neighbors) add(neighbor);
        }
    };
    auto separatedBy(const MergeOp *separator) const -> std::pair<Self, Self>;
    auto isAdjacentTo(const Self& rhs) const -> bool;
    template<std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>, Self>
    static Multiple Combine(R&& neighbors) {
        Multiple result;
        result.add(std::forward<R>(neighbors));
        return result;
    }
    auto isAdjacentTo(const Multiple& rhs) const -> bool;
    auto combinedWith(const Self& rhs) const -> Self;
};

// Canonicalize reshape.
// First we only allow Split's above Merge's,
// then we check for redundant Split's.
// The rule is simple. After the sequence of Merge's, we obtain the smallest reshape blocks,
// and if the blocks that are adjacent get combined by Split's again, this is illegal.
struct ReshapeCanonicalizer: public BottomTopDimVisitor<ReshapeCanonicalizer, ReshapeBlockNeighbors> {
    using Adjacent = ReshapeBlockNeighbors;
    auto transform(const Iterator& dim) const -> Adjacent;
    auto transform(const Reduce& dim) const -> Adjacent;
    auto transform(const RepeatLikeOp& op) const -> Adjacent;
    auto transform(const SplitLikeOp& op) const -> Adjacent;
    auto transform(const MergeLikeOp& op) const -> std::pair<Adjacent, Adjacent>;
};

} // namespace kas
