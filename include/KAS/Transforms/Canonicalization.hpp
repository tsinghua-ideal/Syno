#pragma once

#include "KAS/Core/Graph.hpp"


namespace kas {

// 3 kinds.
// Ordered. Example: H. Merging this with any dimension yields an ordered dimension.
// Unordered, with a specific source. Example: C_in. Merging unordered dimensions from different sources leads to an ordered dimension.
// Unordered, without a specific source. Example: Expand and ShareR. Merging this with an unordered dimension that has a source keeps the source.
struct Unorderedness {
    bool isUnordered;
    std::optional<std::size_t> source;

    static Unorderedness Ordered();
    static Unorderedness Unordered(std::size_t source);
    static Unorderedness Unordered();
    static Unorderedness Unordered(std::optional<std::size_t> source);

    Unorderedness operator&&(const Unorderedness& rhs) const;
};

struct UnorderednessCanonicalizer: public TopBottomDimVisitor<UnorderednessCanonicalizer, Unorderedness> {
    const Graph::DimensionMap<std::size_t>& unorderedDims;
    UnorderednessCanonicalizer(const Graph::DimensionMap<std::size_t>& unorderedDims);
    auto transformInput(const Dimension& dim) -> Unorderedness;
    auto transformExpand(const Dimension& dim) -> Unorderedness;
    auto transform(const RepeatLikeOp& op) -> Unorderedness;
    auto transform(const SplitLikeOp& op) -> std::pair<Unorderedness, Unorderedness>;
    auto transform(const MergeLikeOp& op) -> Unorderedness;
};

// Sometimes we can determine that the unordered dims are from a specific source.
bool IsCanonicalGivenUnorderedness(const Graph& graph, const Graph::DimensionMap<std::size_t>& unorderedDims);

// Weights-sharing can lead to redundancy. We need to check that.
// TODO!!! Check this in ContractionOp::Generate.
bool FinalizationIsCanonicalGivenSharedWeights(const Graph& graph, const std::vector<Topmost>& tensors, const Graph::DimensionSet& sharedWeightDims);

} // namespace kas
