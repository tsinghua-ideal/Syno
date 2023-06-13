#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace kas {

constexpr void HashCombineRaw(std::size_t& seed, std::size_t v) noexcept {
    seed ^= (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template<typename T>
constexpr void HashCombine(std::size_t& seed, const T& v) noexcept {
    std::hash<T> hasher;
    HashCombineRaw(seed, hasher(v));
}

} // namespace kas
