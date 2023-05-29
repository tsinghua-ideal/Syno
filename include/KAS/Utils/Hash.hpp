#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace kas {

template<typename T>
constexpr void HashCombine(std::size_t& seed, const T& v) noexcept {
    if constexpr (std::same_as<std::remove_cvref_t<T>, std::size_t>) {
        seed ^= (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    } else {
        std::hash<T> hasher;
        seed ^= (hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }
}

} // namespace kas
