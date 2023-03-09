#pragma once

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Algorithm.hpp"


namespace kas {

struct ColoredDimension;
struct ColoredInterface;

class Colors {
public:
    enum Color {
        Clear = 0,
        First = 1,
        Second = 2,
        // ...
        Unknown = -1,
    };
    enum class Category {
        Clear,
        Single,
        Unknown,
    };
    struct Options {
        std::size_t maximumTensors = 2;
    };

private:
    const Options& options;

    // The disjoint pairs of set of dimensions, in the sense that they have no common color.
    std::vector<std::pair<std::vector<Dimension>, std::vector<Dimension>>> constraints;

    // Some operations do not require immediate simplification. We use dirty to indicate when we need to simplify.
    // bool dirty = false;

    // When performing simplification, we may encounter inconsistent state. In that case, do not attempt to go any further.
    bool consistent = true;
    void setInconsistent();

public:
    inline Colors(const Options& options): options { options } {}

    void disjoint(const Dimension& lhs, const Dimension& rhs);

    void substitute(ColoredInterface& interface, const Dimension& fro, ColoredDimension to);
    void substitute(ColoredInterface& interface, const Dimension& fro, ColoredDimension to1, ColoredDimension to2);
    void substitute(ColoredInterface& interface, const Dimension& fro1, const Dimension& fro2, ColoredDimension to);
    ColoredDimension& assign(ColoredInterface& interface, const Dimension& item, int color);
    void simplify(ColoredInterface& interface);

    static std::size_t CountColorInconsistent;
    inline bool isConsistent() const { return consistent; }
    bool checkFinalization(const std::vector<Interface>& tensors) const;
};

struct ColoredDimension {
    Dimension dimension;
    // 0 means clear, 1 means the first tensor, 2 means the second tensor, etc.
    // Here we only want to support two tensors at most.
    int color;
    // A color is one of the following: clear, single or unknown.
    inline bool isClear() const noexcept { return color == Colors::Clear; }
    inline bool isSingle() const noexcept { return color > 0; } // Equivalent to not Clear nor Unknown.
    inline bool isUnknown() const noexcept { return color == Colors::Unknown; }
    inline Colors::Category category() const noexcept {
        switch (color) {
        case Colors::Clear:     return Colors::Category::Clear;
        case Colors::Unknown:   return Colors::Category::Unknown;
        default:                return Colors::Category::Single;
        }
    }
    const Size& size() const noexcept { return dimension.size(); }

    struct Projection {
        const Dimension& operator()(const ColoredDimension& item) const noexcept { return item.dimension; }
    };
};

using ColoredInterfaceShapeView = AbstractShape<const std::vector<ColoredDimension>&, [](const ColoredDimension& cDim) -> const Size& { return cDim.dimension.size(); }>;

struct ColoredInterface {
    std::vector<ColoredDimension> items;
    inline std::vector<Dimension> toInterface() const {
        std::vector<Dimension> result;
        result.reserve(items.size());
        std::ranges::copy(items | std::views::transform(ColoredDimension::Projection{}), std::back_inserter(result));
        return result;
    }
    inline std::size_t size() const noexcept { return items.size(); }
    inline ColoredInterfaceShapeView getShape() const { return ColoredInterfaceShapeView(items); }
    inline const Dimension& operator[](std::size_t index) const noexcept { return items[index].dimension; }
    inline const ColoredDimension& operator[](const Dimension& dim) const {
        auto it = binarySearch(dim);
        KAS_ASSERT(it != items.end(), "Dimension not found in interface.");
        return *it;
    }
    inline ColoredDimension& operator[](const Dimension& dim) {
        auto it = binarySearch(dim);
        KAS_ASSERT(it != items.end(), "Dimension not found in interface.");
        return *it;
    }
    inline Dimension& operator[](std::size_t index) noexcept { return items[index].dimension; }
    inline std::vector<ColoredDimension>::const_iterator binarySearch(const Dimension& value) const {
        return WeakOrderedBinarySearch(items, value, Dimension::HashLessThan{}, ColoredDimension::Projection{});
    }
    inline std::vector<ColoredDimension>::iterator binarySearch(const Dimension& value) {
        return WeakOrderedBinarySearch(items, value, Dimension::HashLessThan{}, ColoredDimension::Projection{});
    }
};

} // namespace kas
