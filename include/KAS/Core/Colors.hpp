#pragma once

#include <limits>
#include <vector>

#include "KAS/Utils/Common.hpp"


namespace kas {

class DimensionImpl;
class Dimension;
class MergeLikeOp;

class Color {
    friend class WeightColor;

public:
    // Colors are specified by tags. The disjoint constraints are due to ShareOp's. So in each tag we have a ShareOp, to indicate that such a ShareOp is an acestor of this Dimension.
    using Tag = const MergeLikeOp *;

private:
    // The ancestors ShareOp's of this Op.
    // The tags are sorted.
    std::vector<Tag> tags;
    // Stride discards data.
    bool dataDiscardingFlag = false;
    // Channels are unordered.
    const DimensionImpl *unorderedScope = nullptr;
    // Length of longest chain of primitives below this Dimension.
    int height = 0;
    // Whether all bottom descendants of this dim are Reduce.
    bool endsUpReduceFlag = false;

    Color(auto&& tags, bool dataDiscardingFlag, const DimensionImpl *unorderedScope, int height, bool endsUpReduceFlag):
        tags { std::forward<decltype(tags)>(tags) },
        dataDiscardingFlag { dataDiscardingFlag },
        unorderedScope { unorderedScope },
        height { height },
        endsUpReduceFlag { endsUpReduceFlag }
    {}
    Color(const Color&) = default;

public:
    static bool AnyCommonTags(const std::vector<Tag>& left, const std::vector<Tag>& right);
    static std::vector<Tag> MergeTags(const std::vector<Tag>& left, const std::vector<Tag>& right);
    static bool RemoveTag(std::vector<Tag>& tags, Tag tag);
    // Return the number of tags removed.
    static std::size_t RemoveTags(std::vector<Tag>& tags, const std::vector<Tag>& toRemove);

    Color() = default;
    Color(Color&&) = default;

    // Repeat the color.
    static Color Repeat(const Color& color);
    // Merge two colors, merging the tags, and dataDiscardingFlag. Asserts false on conflict.
    static Color Merge(const Color& lhs, const Color& rhs);

    std::size_t size() const { return tags.size(); }

    // Add a new tag, assuming the op is ShareOp.
    Color& addTag(Tag tag) &;
    Color addTag(Tag tag) && { return std::move(static_cast<Color&>(*this).addTag(tag)); }

    // Returns true if removed.
    bool removeTag(Tag tag);
    bool empty() const { return tags.empty(); }

    bool isDataDiscarding() const { return dataDiscardingFlag; }
    Color& setDataDiscarding(bool value) & { dataDiscardingFlag = value; return *this; }
    Color setDataDiscarding(bool value) && { return std::move(static_cast<Color&>(*this).setDataDiscarding(value)); }
    bool isUnordered() const { return unorderedScope != nullptr; }
    Color& setUnordered(const DimensionImpl *value) &;
    Color setUnordered(const DimensionImpl *value) && { return std::move(static_cast<Color&>(*this).setUnordered(value)); }
    int getHeight() const { return height; }
    Color& setHeight(int value) & { height = value; return *this; }
    Color setHeight(int value) && { return std::move(static_cast<Color&>(*this).setHeight(value)); }
    bool endsUpReduce() const { return endsUpReduceFlag; }
    Color& setEndsUpReduce(bool value) & { endsUpReduceFlag = value; return *this; }
    Color setEndsUpReduce(bool value) && { return std::move(static_cast<Color&>(*this).setEndsUpReduce(value)); }
};

class Graph;

// Different from usual colors, a weight can have left color and right color.
// Left color refers to the left branches of ShareOp, while the right color refers to the right branches.
// Left and right colors must be disjoint to be consistent.
class WeightColor {
    using Tag = Color::Tag;
    std::vector<Tag> leftTags, rightTags;
public:
    WeightColor() = default;
    // Create a basic WeightColor from a ShareR.
    // If the dimension is not a ShareR, then all the tags are seen as left tags.
    WeightColor(const Graph& graph, const Dimension& dim);

    std::size_t countLeftTags() const;
    std::size_t countRightTags() const;
    // Left tags + right tags.
    std::size_t countTags() const;

    // Merge. Asserts that the two are disjoint.
    void merge(const WeightColor& other);

    // Remove all the given right tags from left tags of this.
    void removeAllRightTagsIn(const WeightColor& color);

    // Check for conflicts that will create a bad ShareOp.
    bool disjointWith(const WeightColor& other) const;
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
