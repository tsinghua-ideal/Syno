#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <ranges>
#include <type_traits>
#include <vector>
#include <sstream>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "KAS/Utils/Algorithm.hpp"


namespace kas {

// These `WeakOrderedSubstituteVectorIfAny` first search for the `fro` in `v`. If it found any, it replaces `fro` with `to` and returns `true`. Otherwise, it returns `false`.

template<typename Elem, typename Value, typename ToElem, typename Comp = std::ranges::less, typename Proj = std::identity>
requires std::same_as<std::remove_cvref_t<Elem>, std::remove_cvref_t<ToElem>>
bool WeakOrderedSubstituteVector1To1IfAny(std::vector<Elem>& v, const Value& fro, ToElem&& to, Comp&& comp = {}, Proj&& proj = {}) {
    auto found = WeakOrderedBinarySearch(v, fro, std::forward<Comp>(comp), std::forward<Proj>(proj));
    if (found == v.end()) {
        return false;
    }
    std::vector<Elem> dst;
    dst.reserve(v.size());
    bool toBeInserted = true;
    auto src = v.begin();
    while (src != v.end()) {
        if (src != found) {
            if (toBeInserted && comp(proj(to), proj(*src))) {
                dst.emplace_back(std::forward<ToElem>(to));
                toBeInserted = false;
            }
            dst.emplace_back(std::move(*src));
        }
        ++src;
    }
    if (toBeInserted) {
        dst.emplace_back(std::forward<ToElem>(to));
    }
    v = std::move(dst);
    return true;
}

template<typename Elem, typename Value, typename ToElem, typename Comp = std::ranges::less, typename Proj = std::identity>
requires std::same_as<std::remove_cvref_t<Elem>, std::remove_cvref_t<ToElem>>
bool WeakOrderedSubstituteVector1To2IfAny(std::vector<Elem>& v, const Value& fro, ToElem&& to1, ToElem&& to2, Comp&& comp = {}, Proj&& proj = {}) {
    auto found = WeakOrderedBinarySearch(v, fro, std::forward<Comp>(comp), std::forward<Proj>(proj));
    if (found == v.end()) {
        return false;
    }
    std::vector<Elem> dst;
    dst.reserve(v.size() + 1);
    std::pair<std::reference_wrapper<std::add_const_t<Elem>>, std::reference_wrapper<std::add_const_t<Elem>>> to = comp(proj(to1), proj(to2)) ? std::pair{std::ref(to1), std::ref(to2)} : std::pair{std::ref(to2), std::ref(to1)};
    bool toBeInsertedL = true, toBeInsertedR = true;
    auto src = v.begin();
    while (src != v.end()) {
        if (src != found) {
            if (toBeInsertedL && comp(proj(to.first), proj(*src))) {
                dst.emplace_back(std::forward<ToElem>(to.first.get()));
                toBeInsertedL = false;
            }
            if (toBeInsertedR && comp(proj(to.second), proj(*src))) {
                dst.emplace_back(std::forward<ToElem>(to.second.get()));
                toBeInsertedR = false;
            }
            dst.emplace_back(std::move(*src));
        }
        ++src;
    }
    if (toBeInsertedL) {
        dst.emplace_back(std::forward<ToElem>(to.first));
    }
    if (toBeInsertedR) {
        dst.emplace_back(std::forward<ToElem>(to.second));
    }
    v = std::move(dst);
    return true;
}

template<typename Elem, typename Value, typename ToElem, typename Comp = std::ranges::less, typename Proj = std::identity>
requires std::same_as<std::remove_cvref_t<Elem>, std::remove_cvref_t<ToElem>>
int WeakOrderedSubstituteVector2To1IfAny(std::vector<Elem>& v, const Value& fro1, const Value& fro2, ToElem&& to, Comp&& comp = {}, Proj&& proj = {}) {
    auto found1 = WeakOrderedBinarySearch(v, fro1, std::forward<Comp>(comp), std::forward<Proj>(proj));
    auto found2 = WeakOrderedBinarySearch(v, fro2, std::forward<Comp>(comp), std::forward<Proj>(proj));
    int replaceCount = (found1 != v.end()) + (found2 != v.end());
    if (replaceCount == 0) {
        return 0;
    }
    std::vector<Elem> dst;
    // If both found in `v`, we do not allow duplicates.
    dst.reserve(v.size() + 1 - replaceCount);
    bool toBeInserted = true;
    auto src = v.begin();
    while (src != v.end()) {
        if (src != found1 && src != found2) {
            if (toBeInserted && comp(proj(to), proj(*src))) {
                dst.emplace_back(std::forward<ToElem>(to));
                toBeInserted = false;
            }
            dst.emplace_back(std::move(*src));
        }
        ++src;
    }
    if (toBeInserted) {
        dst.emplace_back(std::forward<ToElem>(to));
    }
    v = std::move(dst);
    return true;
}

template<std::ranges::input_range Range>
std::string VectorToString(Range&& range) {
    return fmt::format("[{}]", fmt::join(range, ", "));
}

} // namespace kas
