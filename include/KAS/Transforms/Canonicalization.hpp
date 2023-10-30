#pragma once

#include "KAS/Core/Graph.hpp"


namespace kas {

// 3 kinds.
// Ordered. Example: H. Merging this with any dimension yields an ordered dimension.
// Unordered, with a specific source. Example: C_in. Merging unordered dimensions from different sources leads to an ordered dimension.
// Unordered, without a specific source. Example: Expand and ShareR. Merging this with an unordered dimension that has a source keeps the source.
struct Unorderedness {
    bool isUnordered;
    std::optional<Dimension> source;

    static Unorderedness Ordered();
    static Unorderedness Unordered(const Dimension& source);
    static Unorderedness Unordered();
    static Unorderedness Unordered(std::optional<Dimension> source);

    Unorderedness operator&&(const Unorderedness& rhs) const;
};

struct UnorderednessCanonicalizer: public TopBottomDimVisitor<UnorderednessCanonicalizer, Unorderedness> {
    const std::set<Dimension, Dimension::AddressLessThan>& unorderedDims;
    UnorderednessCanonicalizer(const std::set<Dimension, Dimension::AddressLessThan>& unorderedDims);
    auto transformInput(const Dimension& dim) -> Unorderedness;
    auto transformExpand(const Dimension& dim) -> Unorderedness;
    auto transform(const RepeatLikeOp& op) -> Unorderedness;
    auto transform(const SplitLikeOp& op) -> std::pair<Unorderedness, Unorderedness>;
    auto transform(const MergeLikeOp& op) -> Unorderedness;
};

bool IsCanonicalGivenUnorderedness(const Graph& graph, const std::set<Dimension, Dimension::AddressLessThan>& unorderedDims);

} // namespace kas
