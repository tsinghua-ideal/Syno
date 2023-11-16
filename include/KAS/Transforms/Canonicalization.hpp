#pragma once

#include "KAS/Core/Graph.hpp"


namespace kas {

// 3 kinds.
// Ordered. Example: H. Merging this with any dimension yields an ordered dimension.
// Unordered, with a specific source. Example: C_in. Merging unordered dimensions from different sources leads to an ordered dimension.
// Unordered, without a specific source. Example: Expand and ShareR. Merging this with an unordered dimension that has a source keeps the source.
struct Unorderedness {
    bool isUnordered = false;
    std::optional<std::size_t> source = std::nullopt;
    const SplitOp *sourceSplitOp = nullptr;

    Unorderedness operator&&(const Unorderedness& rhs) const;
};

struct UnorderednessCanonicalizer: public TopBottomDimVisitor<UnorderednessCanonicalizer, Unorderedness> {
    const Graph::DimensionMap<std::size_t>& unorderedDims;
    bool uncanonical = false;
    UnorderednessCanonicalizer(const Graph::DimensionMap<std::size_t>& unorderedDims);
    auto transformInput(const Dimension& dim) -> Unorderedness;
    auto transformExpand(const Dimension& dim) -> Unorderedness;
    auto transform(const RepeatLikeOp& op) -> Unorderedness;
    auto transform(const SplitLikeOp& op) -> std::pair<Unorderedness, Unorderedness>;
    auto transform(const MergeLikeOp& op) -> Unorderedness;
};

// Sometimes we can determine that the unordered dims are from a specific source.
bool IsCanonicalGivenUnorderedness(const Graph& graph, const Graph::DimensionMap<std::size_t>& unorderedDims);

// True if is pooling.
struct PoolingDiscoverer: public TopBottomDimVisitor<PoolingDiscoverer, bool> {
    auto transformInput(const Dimension& dim) -> bool;
    auto transformExpand(const Dimension& dim) -> bool;
    auto transform(const RepeatLikeOp& op) -> bool;
    auto transform(const SplitLikeOp& op) -> std::pair<bool, bool>;
    auto transform(const MergeLikeOp& op) -> bool;
};

// Doing too much pooling is not great.
bool IsPoolingTooLarge(const Graph& graph, const BindingContext& ctx, std::size_t maxPoolingFactor);

} // namespace kas
