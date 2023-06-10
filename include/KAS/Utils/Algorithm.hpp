#pragma once

#include <algorithm>
#include <compare>
#include <functional>
#include <numeric>
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
    while (left > std::ranges::begin(r) && !(comp(proj(*--left), value))) {
        if (value == proj(*left)) {
            return left;
        }
    }
    return std::ranges::end(r);
}

namespace detail {

template<class F, class T, class I, class U>
concept indirectly_binary_left_foldable_impl =
    std::movable<T> &&
    std::movable<U> &&
    std::convertible_to<T, U> &&
    std::invocable<F&, U, std::iter_reference_t<I>> &&
    std::assignable_from<U&, std::invoke_result_t<F&, U, std::iter_reference_t<I>>>;

template<class F, class T, class I>
concept indirectly_binary_left_foldable =
    std::copy_constructible<F> &&
    std::indirectly_readable<I> &&
    std::invocable<F&, T, std::iter_reference_t<I>> &&
    std::convertible_to<std::invoke_result_t<F&, T, std::iter_reference_t<I>>,
        std::decay_t<std::invoke_result_t<F&, T, std::iter_reference_t<I>>>> &&
    indirectly_binary_left_foldable_impl<F, T, I,
        std::decay_t<std::invoke_result_t<F&, T, std::iter_reference_t<I>>>>;

struct fold_left_first_fn
{
    template<
        std::input_iterator I, std::sentinel_for<I> S,
        indirectly_binary_left_foldable<std::iter_value_t<I>, I> F
    >
    requires
        std::constructible_from<std::iter_value_t<I>, std::iter_reference_t<I>>
    constexpr auto operator()(I first, S last, F f) const
    {
        using U = decltype(
            std::reduce(std::move(first), last, std::iter_value_t<I>(*first), f)
        );
        if (first == last)
            return std::optional<U>();
        std::optional<U> init(std::in_place, *first);
        for (++first; first != last; ++first)
            *init = std::invoke(f, std::move(*init), *first);
        return std::move(init);
    }

    template<
        std::ranges::input_range R,
        indirectly_binary_left_foldable<
            std::ranges::range_value_t<R>, std::ranges::iterator_t<R>
        > F
    >
    requires std::constructible_from<
        std::ranges::range_value_t<R>, std::ranges::range_reference_t<R>
    >
    constexpr auto operator()(R&& r, F f) const
    {
        return (*this)(std::ranges::begin(r), std::ranges::end(r), std::ref(f));
    }
};

} // namespace detail
 
constexpr detail::fold_left_first_fn FoldLeftFirst;

} // namespace kas
