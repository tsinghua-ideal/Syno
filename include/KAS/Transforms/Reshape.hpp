#pragma once

#include "KAS/Core/DimVisitor.hpp"


namespace kas {

struct ReshapeBlockNeighbors;

template<typename T>
concept ReshapeBlockNeighborsRange = std::ranges::input_range<T> && std::same_as<std::ranges::range_value_t<T>, ReshapeBlockNeighbors>;

struct ReshapeBlockNeighbors {
    using Self = ReshapeBlockNeighbors;
    using Side = std::variant<std::monostate, const MergeOp *, const Reduce *>; // nullptr is not allowed.
    struct SidesSet;
    struct Sides {
        Side left, right;
        bool isAdjacentTo(const Sides& rhs) const;
        bool isAdjacentTo(const SidesSet& multiple) const;
        auto separatedBy(const MergeOp *separator) const -> std::pair<Sides, Sides>;
        auto combinedWith(const Sides& rhs) const -> Sides;
    };
    struct ContractedSidesSet;
    struct ContractedSides {
        std::map<int, Sides> sides;
        bool isAdjacentTo(const ContractedSides& rhs) const;
        bool isAdjacentTo(const ContractedSidesSet& multiple) const;
        auto separatedBy(const MergeOp *separator) const -> std::pair<ContractedSides, ContractedSides>;
        auto separatedBy(const ShareOp *separator) const -> std::pair<ContractedSides, ContractedSides>;
        auto combinedWith(const ContractedSides& rhs) const -> ContractedSides;
    };
    ContractedSides sides;
    struct SidesSet {
        std::set<Side> lefts, rights;
        bool hasAdjacent() const;
    };
    struct ContractedSidesSet {
        std::map<int, SidesSet> sides;
        bool hasAdjacent() const;
        void add(const ContractedSides& neighbors);
    };
    struct Multiple {
        ContractedSidesSet sidesSet;
        void add(const ReshapeBlockNeighbors& neighbors);
        template<ReshapeBlockNeighborsRange R>
        void add(R&& neighbors) {
            for (const auto& neighbor: neighbors) add(neighbor);
        }
        bool hasAdjacent() const;
    };
    auto separatedBy(const MergeOp *separator) const -> std::pair<Self, Self>;
    auto separatedBy(const ShareOp *separator) const -> std::pair<Self, Self>;
    auto isAdjacentTo(const Self& rhs) const -> bool;
    template<ReshapeBlockNeighborsRange R>
    static auto Community(R&& neighbors) -> Multiple {
        Multiple result;
        result.add(std::forward<R>(neighbors));
        return result;
    }
    auto isAdjacentTo(const Multiple& rhs) const -> bool;
    static auto AnyAdjacent(const std::vector<Self>& neighbors) -> bool;
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
