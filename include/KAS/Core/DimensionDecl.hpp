#pragma once

#include <boost/container_hash/hash_fwd.hpp>
#include <compare>
#include <initializer_list>
#include <numeric>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class Iterator {
public:
    std::size_t index;
    Size domain;
    explicit inline Iterator(std::size_t index, auto&& domain):
        index { index },
        domain { std::forward<decltype(domain)>(domain) }
    {}
    const Size& size() const noexcept;
};

template<typename T>
concept DimensionLike =
    // is a PrimitiveOp
    PrimitiveOp<T> ||
    // or is an Iterator
    std::same_as<T, Iterator>;

class DimensionImpl;

class Dimension {
public:
    using PointerType = DimensionImpl *;
protected:
    // Require that same `DimensionImpl`s have same address, i.e., uniqued.
    PointerType inner;
public:
    explicit inline Dimension(PointerType inner): inner { inner } {}
    inline PointerType get() const noexcept { return inner; }
    const Size& size() const noexcept;
    // Checks the underlying type of the dimension.
    template<DimensionLike T> bool is() const;
    // Casts the `DimensionImpl` to the desired type.
    template<PrimitiveOp T> const T& as() const;
    bool operator==(const Dimension& other) const = default;
    // Sort the dimensions in an interface to obtain hash for it.
    std::strong_ordering operator<=>(const Dimension& other) const = default;
    std::string description(const BindingContext& ctx) const;
};

} // namespace kas

template<>
struct std::hash<kas::Dimension> {
    inline std::size_t operator()(const kas::Dimension& dimension) const noexcept {
        return std::hash<kas::Dimension::PointerType>{}(dimension.get());
    }
};
