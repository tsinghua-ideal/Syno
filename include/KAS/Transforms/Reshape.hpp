#pragma once

#include "KAS/Core/DimVisitor.hpp"


namespace kas {

struct ReshapeCanonicalizer;

namespace Reshape {

// nullptr is not allowed.
using Side = std::variant<std::monostate, const MergeOp *, const Reduce *>;

struct NeighborsSet;

struct Neighbors {
    Side left, right;
    bool isAdjacentTo(const Neighbors& rhs) const;
    bool isAdjacentTo(const NeighborsSet& multiple) const;
    std::pair<Neighbors, Neighbors> separatedBy(const MergeOp *separator) const;
    Neighbors combinedWith(const Neighbors& rhs) const;
};

using SideSet = std::set<Side>;

struct NeighborsSet {
    SideSet lefts, rights;
};

class Block;

template<typename T>
concept BlockRange = std::ranges::input_range<T> && std::same_as<std::ranges::range_value_t<T>, Block>;

class BlockSet;

class Block {
    friend class BlockSet;
    // The key is contraction id.
    // After going through a Share, primitives applied after the original contraction are propagated independently as well.
    std::map<int, Neighbors> neighborsMap;
public:
    Block();
    Block(const Reduce& dim);
    bool isAdjacentTo(const Block& rhs) const;
    bool isAdjacentTo(const BlockSet& multiple) const;
    std::pair<Block, Block> separatedBy(const MergeOp *separator) const;
    std::pair<Block, Block> separatedBy(const ShareOp *separator) const;
    Block combinedWith(const Block& rhs) const;
    static bool AnyAdjacent(const std::vector<Block>& neighbors);
};

class BlockSet {
    friend class Block;
    std::map<int, NeighborsSet> neighborsMap;
public:
    void add(const Block& block);
    template<BlockRange R>
    void add(R&& neighbors) {
        for (const auto& neighbor: neighbors) add(neighbor);
    }
    template<BlockRange R>
    static BlockSet From(R&& neighbors) {
        BlockSet result;
        result.add(std::forward<R>(neighbors));
        return result;
    }
};

} // namespace Reshape

// Canonicalize reshape.
// First we only allow Split's above Merge's,
// then we check for redundant Split's.
// The rule is simple. After the sequence of Merge's, we obtain the smallest reshape blocks,
// and if the blocks that are adjacent get combined by Split's again, this is illegal.
struct ReshapeCanonicalizer: public BottomTopDimVisitor<ReshapeCanonicalizer, Reshape::Block> {
    auto transform(const Iterator& dim) const -> Reshape::Block;
    auto transform(const Reduce& dim) const -> Reshape::Block;
    auto transform(const RepeatLikeOp& op) const -> Reshape::Block;
    auto transform(const SplitLikeOp& op) const -> Reshape::Block;
    auto transform(const MergeLikeOp& op) const -> std::pair<Reshape::Block, Reshape::Block>;
};

} // namespace kas
