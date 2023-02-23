#pragma once

#include <compare>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Size.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

// The signature of Mapping must be
// (const std::remove_cvref_t<Storage>::value_type&) -> const Size&
template<typename Storage, auto Mapping>
class AbstractShape {
    Storage sizes;

    struct Iterator {
        using underlying = typename std::remove_cvref_t<Storage>::const_iterator;

        using value_type = const Size;
        using difference_type = typename underlying::difference_type;
        using pointer = value_type *;
        using reference = value_type&;

        underlying dim;

        inline Iterator(): dim {} {}
        inline Iterator(underlying dim): dim { dim } {}
        inline Iterator(const Iterator& other): dim { other.dim } {}

        inline reference operator*() const { return Mapping(*dim); }
        inline pointer operator->() const { return &Mapping(*dim); }
        inline reference operator[](std::size_t n) const { return Mapping(dim[n]); }

        inline Iterator& operator++() { ++dim; return *this; }
        inline Iterator operator++(int) { auto res = *this; ++dim; return res; }
        inline Iterator& operator--() { --dim; return *this; }
        inline Iterator operator--(int) { auto res = *this; --dim; return res; }
        inline Iterator& operator+=(difference_type n) { dim += n; return *this; }
        inline Iterator& operator-=(difference_type n) { dim -= n; return *this; }
        inline Iterator operator+(difference_type n) const { return dim + n; }
        friend inline Iterator operator+(difference_type n, const Iterator& it) { return n + it.dim; }
        inline Iterator operator-(difference_type n) const { return dim - n; }
        inline difference_type operator-(const Iterator& other) const { return dim - other.dim; }

        inline bool operator==(const Iterator& other) const = default;
        inline std::strong_ordering operator<=>(const Iterator& other) const = default;
    };

public:
    AbstractShape(auto&& sizes): sizes { std::forward<decltype(sizes)>(sizes) } {}

    inline std::size_t size() const { return sizes.size(); }
    inline const Size& operator[](std::size_t i) const { return Mapping(sizes[i]); }

    inline Iterator begin() const { return Iterator { sizes.begin() }; }
    inline Iterator end() const { return Iterator { sizes.end() }; }

    bool operator==(const Shape& other) const;

    inline Size totalSize() const {
        return Size::Product(*this);
    }

    template<typename ValueType, typename Tp, typename Tc>
    std::vector<ValueType> eval(Tp&& p, Tc&& c) const {
        std::vector<ValueType> result;
        for (const auto& size: *this) {
            result.emplace_back(size.template eval<ValueType>(std::forward<Tp>(p), std::forward<Tc>(c)));
        }
        return result;
    };

    inline std::vector<std::size_t> estimate(const BindingContext& ctx) const {
        std::vector<std::size_t> result;
        for (const auto& size: sizes) {
            result.emplace_back(size.estimate(ctx));
        }
        return result;
    }

    inline std::string toString(const BindingContext& ctx) const {
        return VectorToString(*this | std::views::transform([&ctx](const Size& size) {
            return size.toString(ctx);
        }));
    }
};

// We have forward-defined Shape in BindingContext.hpp.
// using Shape = AbstractShape<std::vector<Size>, [](const Size& size) -> const Size& { return size; }>;
using ShapeView = AbstractShape<const std::vector<Dimension>&, [](const Dimension& dim) -> const Size& { return dim.size(); }>;

} // namespace kas
