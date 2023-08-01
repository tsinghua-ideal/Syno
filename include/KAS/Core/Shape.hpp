#pragma once

#include <compare>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Size.hpp"
#include "KAS/Utils/Vector.hpp"


namespace kas {

// The signature of Mapping must be
// (const std::remove_cvref_t<Storage>::value_type&) -> const Size&
template<typename Storage, auto Mapping>
class AbstractShape {
public:
    Storage sizes;

    struct Iterator {
        using underlying = typename std::remove_cvref_t<Storage>::const_iterator;

        using value_type = const Size;
        using difference_type = typename underlying::difference_type;
        using pointer = value_type *;
        using reference = value_type&;

        underlying dim;

        Iterator(): dim {} {}
        Iterator(underlying dim): dim { dim } {}
        Iterator(const Iterator& other): dim { other.dim } {}

        reference operator*() const { return Mapping(*dim); }
        pointer operator->() const { return &Mapping(*dim); }
        reference operator[](std::size_t n) const { return Mapping(dim[n]); }

        Iterator& operator++() { ++dim; return *this; }
        Iterator operator++(int) { auto res = *this; ++dim; return res; }
        Iterator& operator--() { --dim; return *this; }
        Iterator operator--(int) { auto res = *this; --dim; return res; }
        Iterator& operator+=(difference_type n) { dim += n; return *this; }
        Iterator& operator-=(difference_type n) { dim -= n; return *this; }
        Iterator operator+(difference_type n) const { return dim + n; }
        friend Iterator operator+(difference_type n, const Iterator& it) { return n + it.dim; }
        Iterator operator-(difference_type n) const { return dim - n; }
        difference_type operator-(const Iterator& other) const { return dim - other.dim; }

        bool operator==(const Iterator& other) const = default;
        std::strong_ordering operator<=>(const Iterator& other) const = default;
    };

    AbstractShape() requires(std::is_default_constructible_v<Storage>) = default;
    AbstractShape(auto&& sizes): sizes { std::forward<decltype(sizes)>(sizes) } {}

    std::size_t size() const { return sizes.size(); }
    const Size& operator[](std::size_t i) const { return Mapping(sizes[i]); }

    Iterator begin() const { return Iterator { sizes.begin() }; }
    Iterator end() const { return Iterator { sizes.end() }; }

    Size totalSize() const {
        return Size::Product(*this);
    }

    template<typename ValueType>
    std::vector<ValueType> eval(const ConcreteConsts& consts) const {
        std::vector<ValueType> result;
        for (const auto& size: *this) {
            result.emplace_back(size.template eval<ValueType>(consts));
        }
        return result;
    }

    std::string toString(const BindingContext& ctx) const {
        return VectorToString(*this | std::views::transform([&ctx](const Size& size) {
            return size.toString(ctx);
        }));
    }

    // FOR DEBUG USAGE ONLY!
    std::string debugToString() const {
        return BindingContext::ApplyDebugPublicCtx(&std::remove_cvref_t<decltype(*this)>::toString, *this);
    }
};

// We have forward-defined Shape in BindingContext.hpp.
// using Shape = AbstractShape<std::vector<Size>, [](const Size& size) -> const Size& { return size; }>;

} // namespace kas
