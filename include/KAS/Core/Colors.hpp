#pragma once

#include <limits>
#include <vector>

#include "KAS/Utils/Common.hpp"


namespace kas {

class Dimension;
class MergeLikeOp;

class Color {
public:
    // Colors are specified by tags. The disjoint constraints are due to ShareOp's. So in each tag we have a ShareOp, to indicate that such a ShareOp is an acestor of this Dimension.
    using Tag = const MergeLikeOp *;

private:
    // The ancestors ShareOp's of this Op.
    // The tags are sorted.
    std::vector<Tag> tags;
    bool dataDiscardingFlag = false;

public:
    Color() = default;
    Color(auto&& color1, auto&&color2):
        Color { std::forward<decltype(color1)>(color1) }
    {
        merge(std::forward<decltype(color2)>(color2));
    }
    std::size_t size() const { return tags.size(); }

    // Merge two colors, merging the tags, and dataDiscardingFlag. Asserts false on conflict.
    void merge(const Color& other);

    // Add a new tag, assuming the op is ShareOp.
    void addTag(Tag tag);

    bool isDataDiscarding() const { return dataDiscardingFlag; }
    void setDataDiscarding(bool value) { dataDiscardingFlag = value; }

    // Assume the dimension is a ShareR or Iterator. Rejects if the Op of the ShareR is in the tags.
    bool disjointWithWeightDim(const Dimension& dim) const;
    // Assume the dimension is a ShareR or Iterator. Remove the Op of the ShareR from its color if any, and merge the color.
    void mergeWeightDim(const Dimension& dim);

    // Returns true if removed.
    bool removeTag(Tag tag);
    bool empty() const { return tags.empty(); }

    static const Color None;

    // This assigns the dimensions with colors, and verify that color constraints are not violated. 
    // For correctly constructed Finalizations, this is intended to return true.
    static bool CheckFinalization(const std::vector<std::vector<Dimension>>& tensors);
};

// Use bits to represent colors.
class CompactColor {
    std::size_t value;
public:
    explicit CompactColor(std::size_t value): value { value } {}
    explicit CompactColor(int single): value { std::size_t{1} << single } {
        KAS_ASSERT(single >= 0 && single < std::numeric_limits<std::size_t>::digits);
    }
    // Intersection.
    CompactColor operator&(const CompactColor& rhs) const noexcept { return CompactColor { value & rhs.value }; }
    // Union.
    CompactColor operator|(const CompactColor& rhs) const noexcept { return CompactColor { value | rhs.value }; }
    explicit operator bool() const noexcept { return value != 0; }
};

} // namespace kas
