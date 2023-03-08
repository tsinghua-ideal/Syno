#pragma once

#include <algorithm>
#include <compare>
#include <ranges>


namespace kas {

template<std::ranges::random_access_range R, typename Comp = std::ranges::less, typename Proj = std::identity>
auto WeakOrderedBinarySearch(R&& r, const auto& value, Comp&& comp = {}, Proj&& proj = {}) -> std::ranges::iterator_t<R> {
    auto begin = std::ranges::begin(r);
    auto end = std::ranges::end(r);
    auto size = std::ranges::size(r);
    auto mid = begin + size / 2;
    while (size > 0) {
        if (comp(value, proj(*mid))) {
            end = mid;
        } else if (comp(proj(*mid), value)) {
            begin = mid + 1;
        } else {
            break;
        }
        size = end - begin;
        mid = begin + size / 2;
    }
    if (mid == end) {
        return std::ranges::end(r);
    }
    // Search right and left to find an equal element.
    auto right = mid;
    while (right < std::ranges::end(r) && !(comp(value, proj(*right)))) {
        if (value == proj(*right)) {
            return right;
        }
        ++right;
    }
    auto left = mid;
    while (left > std::ranges::begin(r)) {
        if (value == proj(*--left)) {
            return left;
        }
    }
    return std::ranges::end(r);
}

} // namespace kas
