#pragma once

#include <compare>
#include <concepts>
#include <cstddef>
#include <ranges>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Size.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

class Shape {
    std::vector<Size> sizes;

    struct Iterator {
        using value_type = const Size;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type *;
        using reference = value_type&;

        pointer dim;

        inline Iterator(): dim { nullptr } {}
        inline Iterator(pointer dim): dim { dim } {}
        inline Iterator(const Iterator& other): dim { other.dim } {}

        inline reference operator*() const { return *dim; }
        inline pointer operator->() const { return dim; }
        inline reference operator[](difference_type n) const { return dim[n]; }

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
    Shape(auto&& sizes): sizes { std::forward<decltype(sizes)>(sizes) } {}

    inline std::size_t size() const { return sizes.size(); }
    inline const Size& operator[](std::size_t i) const { return sizes[i]; }

    inline Iterator begin() const { return Iterator { sizes.data() }; }
    inline Iterator end() const { return Iterator { sizes.data() + sizes.size() }; }

    bool operator==(const Shape& other) const;

    inline Size totalSize() const {
        return Size::Product(*this);
    }

    template<typename ValueType, typename Tp, typename Tc>
    std::vector<ValueType> eval(Tp&& p, Tc&& c) const {
        std::vector<ValueType> result;
        for (const auto& size: *this) {
            result.emplace_back(size.eval<ValueType>(std::forward<Tp>(p), std::forward<Tc>(c)));
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
        return VectorToString(*this | std::ranges::views::transform([&ctx](const Size& size) {
            return size.toString(ctx);
        }));
    }
};

class ShapeView {
    const Interface& sizes;

    struct Iterator {
        using value_type = const Size;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type *;
        using reference = value_type&;

        const Dimension *dim;

        inline Iterator(): dim { nullptr } {}
        inline Iterator(const Dimension *dim): dim { dim } {}
        inline Iterator(const Iterator& other): dim { other.dim } {}

        inline reference operator*() const { return dim->size(); }
        inline pointer operator->() const { return &dim->size(); }
        inline reference operator[](difference_type n) const { return dim[n].size(); }

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
    static_assert(std::random_access_iterator<Iterator>);

public:
    inline ShapeView(const Interface& sizes):
        sizes { sizes }
    {}

    inline std::size_t size() const { return sizes.size(); }
    inline const Size& operator[](std::size_t index) const { return sizes[index].size(); }

    inline Iterator begin() const { return sizes.data(); }
    inline Iterator end() const { return sizes.data() + sizes.size(); }

    bool operator==(const ShapeView& other) const;

    Size totalSize() const;

    template<typename ValueType, typename Tp, typename Tc>
    std::vector<ValueType> eval(Tp&& p, Tc&& c) const {
        std::vector<ValueType> result;
        for (const auto& size: *this) {
            result.emplace_back(size.eval<ValueType>(std::forward<Tp>(p), std::forward<Tc>(c)));
        }
        return result;
    };

    std::vector<std::size_t> estimate(const BindingContext& ctx) const;

    std::string toString(const BindingContext& ctx) const;
};

} // namespace kas
