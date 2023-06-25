#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <type_traits>

namespace kas {

template<std::size_t Range>
constexpr std::size_t HashForSmallRange(std::size_t v) noexcept {
    // We require that v < Range.
    constexpr int SizeTypeWidth = std::numeric_limits<std::size_t>::digits;
    constexpr int Shift = SizeTypeWidth / static_cast<int>(Range);
    constexpr std::size_t Mask = (std::size_t(1) << Shift) - 1;
    return Mask << (v * Shift);
}

constexpr void HashCombineRaw(std::size_t& seed, std::size_t v) noexcept {
    seed ^= (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template<typename T>
constexpr void HashCombine(std::size_t& seed, const T& v) noexcept {
    std::hash<T> hasher;
    HashCombineRaw(seed, hasher(v));
}

} // namespace kas
