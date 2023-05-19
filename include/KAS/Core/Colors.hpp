#pragma once

#include <compare>
#include <limits>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Algorithm.hpp"


namespace kas {

class MergeLikeOp;
struct ColoredDimension;
class ColoredInterface;

class Color {
public:
    // Colors are specified by tags. The disjoint constraints are due to ShareOp's. So in each tag we have a ShareOp, to indicate that such a ShareOp is an acestor of this Dimension.
    struct Tag {
        // Left or right.
        Order order;
        // The ShareOp.
        const MergeLikeOp *share;

        // To sort the tags.
        std::strong_ordering operator<=>(const Tag& rhs) const noexcept = default;
    };

private:
    // Keep them sorted! Note that tags.size() <= maximumTensors - 1.
    std::vector<Tag> tags;

public:
    // With no tags.
    Color() = default;
    // Copy the color.
    Color(const Color&) = default;
    // Add a new tag. Assumes the dim is a ShareOp::Input.
    Color(const Color& color, const Dimension& dim);
    // Merge two colors, merging the tags.
    Color(const Color& lhs, const Color& rhs);

    std::size_t countTags() const noexcept { return tags.size(); }

    // Whether there are no common tags.
    bool disjoint(const Color& rhs) const;
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

struct ColoredDimension {
    Dimension dimension;
    Color color;

    const Size& size() const noexcept { return dimension.size(); }

    struct Projection {
        const Dimension& operator()(const ColoredDimension& item) const noexcept { return item.dimension; }
    };
};

using ColoredInterfaceShapeView = AbstractShape<const std::vector<ColoredDimension>&, [](const ColoredDimension& cDim) -> const Size& { return cDim.dimension.size(); }>;
class Graph;

class ColoredInterface {
    std::vector<ColoredDimension> items;
    std::size_t maximumTags;

public:
    template<std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, ColoredDimension>
    ColoredInterface(R&& r):
        items { std::forward<R>(r) },
        maximumTags { std::ranges::max(items, {}, [](auto&& cdim) { return cdim.color.countTags(); }) }
    {}

    template<DimensionRange R>
    ColoredInterface(R&& r): maximumTags { 0 } {
        items.reserve(std::ranges::size(r));
        for (auto&& dim: r) {
            items.emplace_back(dim, Color {});
        }
    }

    auto toDimensions() const {
        return items | std::views::transform(&ColoredDimension::dimension);
    }
    std::size_t size() const noexcept { return items.size(); }
    ColoredInterfaceShapeView getShape() const { return ColoredInterfaceShapeView(items); }

    auto begin() const { return items.begin(); }
    auto begin() { return items.begin(); }
    auto end() const { return items.end(); }
    auto end() { return items.end(); }
    const Dimension& operator[](std::size_t index) const noexcept { return items[index].dimension; }
    const ColoredDimension& operator[](const Dimension& dim) const {
        auto it = binarySearch(dim);
        KAS_ASSERT(it != items.end(), "Dimension not found in interface.");
        return *it;
    }
    ColoredDimension& operator[](const Dimension& dim) {
        return const_cast<ColoredDimension&>(const_cast<const ColoredInterface&>(*this)[dim]);
    }
    Dimension& operator[](std::size_t index) noexcept { return items[index].dimension; }
    std::vector<ColoredDimension>::const_iterator binarySearch(const Dimension& value) const {
        return WeakOrderedBinarySearch(items, value, Dimension::HashLessThan{}, ColoredDimension::Projection{});
    }
    std::vector<ColoredDimension>::iterator binarySearch(const Dimension& value) {
        auto it = const_cast<const ColoredInterface&>(*this).binarySearch(value);
        // Amazing trick: https://stackoverflow.com/questions/765148/how-to-remove-constness-of-const-iterator
        return items.erase(it, it);
    }

    ColoredInterface substitute1to1(const Dimension& fro, const Dimension& to) const;
    ColoredInterface substitute1to2(const Dimension& fro, const Dimension& to1, const Dimension& to2, bool addConstraint = false) const;
    ColoredInterface substitute2to1(const Dimension& fro1, const Dimension& fro2, const Dimension& to) const;

    Graph buildGraph() const;

    auto filterOut(const std::vector<DimensionTypeWithOrder>& disallows) const {
        return items | std::views::filter([&](const ColoredDimension& cdim) {
            return std::ranges::none_of(disallows, [&](auto disallow) { return cdim.dimension.is(disallow); });
        });
    }
};

} // namespace kas
