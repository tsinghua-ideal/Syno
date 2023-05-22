#pragma once

/*
This code has been adapted from rangesnext
https://github.com/cor3ntin/rangesnext

Licenced under Boost Software License license. See LICENSE.txt for details.
*/

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <ranges>
#include <tuple>
#include <utility>


namespace kas::ranges::detail {

namespace r = std::ranges;

template <typename T>
concept has_iterator_category = requires {
    typename std::iterator_traits<r::iterator_t<T>>::iterator_category;
};

template <typename T>
concept has_iterator_concept = requires {
    typename std::iterator_traits<r::iterator_t<T>>::iterator_concept;
};

template <typename... V>
consteval auto iter_cat() {
    if constexpr ((r::random_access_range<V> && ...))
        return std::random_access_iterator_tag{};
    else if constexpr ((r::bidirectional_range<V> && ...))
        return std::bidirectional_iterator_tag{};
    else if constexpr ((r::forward_range<V> && ...))
        return std::forward_iterator_tag{};
    else if constexpr ((r::input_range<V> && ...))
        return std::input_iterator_tag{};
    else
        return std::output_iterator_tag{};
}

template <class R>
concept simple_view = r::view<R> &&r::range<const R>
    &&std::same_as<r::iterator_t<R>, r::iterator_t<const R>>
        &&std::same_as<r::sentinel_t<R>, r::sentinel_t<const R>>;

template <typename T>
inline constexpr bool pair_like = false;
template <typename F, typename S>
inline constexpr bool pair_like<std::pair<F, S>> = true;
template <typename F, typename S>
inline constexpr bool pair_like<std::tuple<F, S>> = true;
} // namespace kas::ranges::detail

namespace kas::ranges {

namespace r = std::ranges;

template <class R>
concept simple_view = // exposition only
    r::view<R> &&r::range<const R>
        &&std::same_as<r::iterator_t<R>, r::iterator_t<const R>>
            &&std::same_as<r::sentinel_t<R>, r::sentinel_t<const R>>;

template <r::input_range V>
requires r::view<V> class enumerate_view
    : public r::view_interface<enumerate_view<V>> {

    V base_ = {};

    template <bool>
    struct sentinel;

    template <bool Const>
    struct iterator {
      private:
        using Base = std::conditional_t<Const, const V, V>;
        using count_type = decltype([] {
            if constexpr (r::sized_range<Base>)
                return r::range_size_t<Base>();
            else {
                return std::make_unsigned_t<r::range_difference_t<Base>>();
            }
        }());

        template <typename T>
        struct result {
            const count_type index;
            T value;

            constexpr bool operator==(const result &other) const = default;
        };

        r::iterator_t<Base> current_ = r::iterator_t<Base>();
        count_type pos_ = 0;

        template <bool>
        friend struct iterator;
        template <bool>
        friend struct sentinel;

      public:
        using iterator_category = decltype(detail::iter_cat<Base>());
        using reference = result<r::range_reference_t<Base>>;
        using value_type = result<r::range_reference_t<Base>>;
        using difference_type = r::range_difference_t<Base>;

        iterator() = default;

        constexpr explicit iterator(r::iterator_t<Base> current,
                                    r::range_difference_t<Base> pos)
            : current_(std::move(current)), pos_(pos) {
        }
        constexpr iterator(iterator<!Const> i) requires Const
            &&std::convertible_to<r::iterator_t<V>, r::iterator_t<Base>>
            : current_(std::move(i.current_)), pos_(i.pos_) {
        }

        constexpr const r::iterator_t<V> &
        base() const &requires std::copyable<r::iterator_t<Base>> {
            return current_;
        }

        constexpr r::iterator_t<V> base() && {
            return std::move(current_);
        }

        constexpr auto operator*() const {
            return reference{static_cast<count_type>(pos_), *current_};
        }

        constexpr iterator &operator++() {
            ++pos_;
            ++current_;
            return *this;
        }

        constexpr auto operator++(int) {
            ++pos_;
            if constexpr (r::forward_range<V>) {
                auto tmp = *this;
                ++*this;
                return tmp;
            } else {
                ++current_;
            }
        }

        constexpr iterator &operator--() requires r::bidirectional_range<V> {
            --pos_;
            --current_;
            return *this;
        }

        constexpr auto operator--(int) requires r::bidirectional_range<V> {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr iterator &
        operator+=(difference_type n) requires r::random_access_range<V> {
            current_ += n;
            pos_ += n;
            return *this;
        }

        constexpr iterator &
        operator-=(difference_type n) requires r::random_access_range<V> {
            current_ -= n;
            pos_ -= n;
            return *this;
        }

        friend constexpr iterator
        operator+(const iterator &i,
                  difference_type n) requires r::random_access_range<V> {
            return iterator{i.current_ + n,
                            static_cast<difference_type>(i.pos_ + n)};
        }

        friend constexpr iterator
        operator+(difference_type n,
                  const iterator &i) requires r::random_access_range<V> {
            return iterator{i.current_ + n,
                            static_cast<difference_type>(i.pos_ + n)};
        }

        friend constexpr auto
        operator-(iterator i,
                  difference_type n) requires r::random_access_range<V> {
            return iterator{i.current_ - n,
                            static_cast<difference_type>(i.pos_ - n)};
        }

        friend constexpr auto
        operator-(difference_type n,
                  iterator i) requires r::random_access_range<V> {
            return iterator{i.current_ - n,
                            static_cast<difference_type>(i.pos_ - n)};
        }

        constexpr decltype(auto) operator[](difference_type n) const
            requires r::random_access_range<Base> {
            return reference{static_cast<count_type>(pos_ + n),
                             *(current_ + n)};
        }

        friend constexpr bool operator==(
            const iterator &x,
            const iterator
                &y) requires std::equality_comparable<r::iterator_t<Base>> {
            return x.current_ == y.current_;
        }

        template <bool ConstS>
        friend constexpr bool operator==(const iterator<Const> &i,
                                         const sentinel<ConstS> &s) {
            return i.current_ == s.base();
        }

        friend constexpr bool
        operator<(const iterator &x,
                  const iterator &y) requires r::random_access_range<Base> {
            return x.current_ < y.current_;
        }

        friend constexpr bool
        operator>(const iterator &x,
                  const iterator &y) requires r::random_access_range<Base> {
            return x.current_ > y.current_;
        }

        friend constexpr bool
        operator<=(const iterator &x,
                   const iterator &y) requires r::random_access_range<Base> {
            return x.current_ <= y.current_;
        }
        friend constexpr bool
        operator>=(const iterator &x,
                   const iterator &y) requires r::random_access_range<Base> {
            return x.current_ >= y.current_;
        }
        friend constexpr auto
        operator<=>(const iterator &x,
                    const iterator &y) requires r::random_access_range<Base>
            &&std::three_way_comparable<r::iterator_t<Base>> {
            return x.current_ <=> y.current_;
        }

        friend constexpr difference_type
        operator-(const iterator &x,
                  const iterator &y) requires r::random_access_range<Base> {
            return x.current_ - y.current_;
        }
    };

    template <bool Const>
    struct sentinel {
      private:
        friend iterator<false>;
        friend iterator<true>;

        using Base = std::conditional_t<Const, const V, V>;

        r::sentinel_t<V> end_;

      public:
        sentinel() = default;

        constexpr explicit sentinel(r::sentinel_t<V> end)
            : end_(std::move(end)) {
        }

        constexpr auto base() const {
            return end_;
        }

        friend constexpr r::range_difference_t<Base>
        operator-(const iterator<Const> &x, const sentinel &y) requires std::
            sized_sentinel_for<r::sentinel_t<Base>, r::iterator_t<Base>> {
            return x.current_ - y.end_;
        }

        friend constexpr r::range_difference_t<Base>
        operator-(const sentinel &x, const iterator<Const> &y) requires std::
            sized_sentinel_for<r::sentinel_t<Base>, r::iterator_t<Base>> {
            return x.end_ - y.current_;
        }
    };

  public:
    constexpr enumerate_view() = default;
    constexpr enumerate_view(V base) : base_(std::move(base)) {
    }

    constexpr auto begin() requires(!simple_view<V>) {
        return iterator<false>(std::ranges::begin(base_), 0);
    }

    constexpr auto begin() const requires simple_view<V> {
        return iterator<true>(std::ranges::begin(base_), 0);
    }

    constexpr auto end() {
        return sentinel<false>{r::end(base_)};
    }

    constexpr auto end() requires r::common_range<V> {
        return iterator<false>{std::ranges::end(base_),
                               static_cast<r::range_difference_t<V>>(size())};
    }

    constexpr auto end() const requires r::range<const V> {
        return sentinel<true>{std::ranges::end(base_)};
    }

    constexpr auto end() const requires r::common_range<const V> {
        return iterator<true>{std::ranges::end(base_),
                              static_cast<r::range_difference_t<V>>(size())};
    }

    constexpr auto size() requires r::sized_range<V> {
        return std::ranges::size(base_);
    }

    constexpr auto size() const requires r::sized_range<const V> {
        return std::ranges::size(base_);
    }

    constexpr V base() const &requires std::copyable<V> {
        return base_;
    }

    constexpr V base() && {
        return std::move(base_);
    }
};

template <typename R>
requires r::input_range<R> enumerate_view(R &&r)
    -> enumerate_view<r::views::all_t<R>>;

namespace detail {

struct enumerate_view_fn {
    template <typename R>
    constexpr auto operator()(R &&r) const {
        return enumerate_view{std::forward<R>(r)};
    }

    template <r::input_range R>
    constexpr friend auto operator|(R &&rng, const enumerate_view_fn &) {
        return enumerate_view{std::forward<R>(rng)};
    }
};
} // namespace detail

inline detail::enumerate_view_fn enumerate;

} // namespace kas::ranges

namespace kas::ranges {

struct from_range_t {};
inline constexpr from_range_t from_range;

namespace detail {

namespace r = std::ranges;

template <typename T>
inline constexpr bool always_false_v = false;

template <r::range Rng>
struct range_common_iterator_impl {
    using type = std::common_iterator<std::ranges::iterator_t<Rng>, r::sentinel_t<Rng>>;
};

template <typename Rng>
struct dummy_input_iterator {
    using iterator_category = std::input_iterator_tag;
    using value_type = std::ranges::range_value_t<Rng>;
    using difference_type = std::ranges::range_difference_t<Rng>;
    ;
    using pointer = std::ranges::range_value_t<Rng> *;
    using reference = std::ranges::range_reference_t<Rng>;

    int operator*() const;
    bool operator==(const dummy_input_iterator &other) const;
    reference operator++(int);
    dummy_input_iterator &operator++();
};

template <r::common_range Rng>
struct range_common_iterator_impl<Rng> {
    using type = r::iterator_t<Rng>;
};

template <r::range Rng>
requires(!std::copyable<std::ranges::iterator_t<Rng>>) struct range_common_iterator_impl<Rng> {
    using type = dummy_input_iterator<Rng>;
};

template <r::range Rng>
using range_common_iterator = typename range_common_iterator_impl<Rng>::type;

template <typename C>
struct container_value;

template <typename C>
requires requires {
    typename r::range_value_t<C>;
}
struct container_value<C> {
    using type = r::range_value_t<C>;
};
template <typename C>
requires(!requires { typename r::range_value_t<C>; }) struct container_value<C> {
    using type = typename C::value_type;
};

template <typename C>
using container_value_t = container_value<C>::type;

template <class C, class R>
concept container_convertible =
    !r::view<C> && r::input_range<R> && std::convertible_to<r::range_reference_t<R>, container_value_t<C>>;

template <class C, class R>
concept recursive_container_convertible = container_convertible<C, R> ||
    (r::input_range<r::range_reference_t<R>> &&requires {
        {
            to<r::range_value_t<C>>(std::declval<r::range_reference_t<R>>())
            } -> std::convertible_to<r::range_value_t<C>>;
    });

} // namespace detail

template <typename Cont, std::ranges::input_range Rng, typename... Args>
requires detail::recursive_container_convertible<Cont, Rng>
constexpr auto to(Rng &&rng, Args &&...args) -> Cont;

namespace detail {

template <template <class...> class T>
struct wrap {};

template <typename Cont, typename Rng, typename... Args>
struct unwrap {
    using type = Cont;
};

template <template <class...> class Cont, typename Rng, typename... Args>
struct unwrap<wrap<Cont>, Rng, Args...> {
    template <typename R>
    static auto from_rng(int)
        -> decltype(Cont(range_common_iterator<Rng>(), range_common_iterator<Rng>(), std::declval<Args>()...));
    template <typename R>
    static auto from_rng(long) -> decltype(Cont(from_range, std::declval<Rng>(), std::declval<Args>()...));

    using type = std::remove_cvref_t<std::remove_pointer_t<decltype(from_rng<Rng>(0))>>;
};

template <typename T>
concept reservable_container = requires(T &c) {
    c.reserve(r::size(c));
    {c.capacity()} -> std::same_as<decltype(r::size(c))>;
    {c.max_size()} -> std::same_as<decltype(r::size(c))>;
};

template <typename T>
concept insertable_container = requires(T &c, T::value_type &e) {
    c.insert(c.end(), e);
};

struct to_container {
  private:
    template <typename ToContainer, typename Rng, typename... Args>
    using container_t = typename unwrap<ToContainer, Rng, Args...>::type;

    template <typename C, typename... Args>
    struct fn {
      private:
        template <typename Cont, typename Rng>
        constexpr static auto construct(Rng &&rng, Args &&...args) {
            auto inserter = [](Cont &c) {
                if constexpr (requires { c.push_back(std::declval<std::ranges::range_reference_t<Rng>>()); }) {
                    return std::back_inserter(c);
                } else {
                    return std::inserter(c, std::end(c));
                }
            };

            // copy or move (optimization)
            if constexpr (std::constructible_from<Cont, Rng, Args...>) {
                return Cont(std::forward<Rng>(rng), std::forward<Args>(args)...);
            } else if constexpr (std::constructible_from<Cont, from_range_t, Rng, Args...>) {
                return Cont(from_range, std::forward<Rng>(rng), std::forward<Args>(args)...);
            }
            else if constexpr (r::common_range<Rng> && std::constructible_from<Cont, r::iterator_t<Rng>, r::iterator_t<Rng>, Args...>) {
                return Cont(r::begin(rng), r::end(rng), std::forward<Args>(args)...);
            }
            // we can do push back
            else if constexpr (insertable_container<Cont> &&
                               std::constructible_from<Cont, Args...>) {
                Cont c(std::forward<Args>(args)...);
                if constexpr(r::sized_range<Rng> && reservable_container<Cont>) {
                    c.reserve(r::size(rng));
                }
                r::copy(std::forward<Rng>(rng), inserter(c));
                return c;
            } else {
                static_assert(always_false_v<Cont>, "Can't construct a container");
            }
        }

        template <typename Cont, r::range Rng>
        requires container_convertible<Cont, Rng>
        constexpr static auto impl(Rng &&rng, Args &&...args) {
            return construct<Cont>(std::forward<Rng>(rng), std::forward<Args>(args)...);
        }

        template <typename Cont, r::range Rng>
        requires recursive_container_convertible<Cont, Rng> && std::constructible_from<Cont, Args...> &&
            (!container_convertible<Cont, Rng> &&
             !std::constructible_from<Cont, Rng>)constexpr static auto impl(Rng &&rng, Args &&...args) {

            return to<Cont, Args...>(rng | r::views::transform([](auto &&elem)
            { return to<r::range_value_t<Cont>>(std::forward<decltype(elem)>(elem)); }), std::forward<Args>(args)...);
        }

      public:
        template <typename Rng>
        requires r::input_range<Rng> &&
            recursive_container_convertible<container_t<C, Rng, Args...>, Rng &&> inline constexpr auto
            operator()(Rng &&rng, Args &&...args) const {
            return impl<container_t<C, Rng, Args...>>(std::forward<Rng>(rng), std::forward<Args>(args)...);
        }
        std::tuple<Args...> args;
    };

    template <typename Rng, typename ToContainer, typename... Args>
    requires r::input_range<Rng> && recursive_container_convertible<container_t<ToContainer, Rng, Args...>, Rng>
    constexpr friend auto operator|(Rng &&rng, fn<ToContainer, Args...> &&f) -> container_t<ToContainer, Rng, Args...> {
        return [&]<size_t... I>(std::index_sequence<I...>) {
            return f(std::forward<Rng>(rng), std::forward<Args>(std::get<I>(f.args))...);
        }
        (std::make_index_sequence<sizeof...(Args)>());
    }
};

template <typename ToContainer, typename... Args>
using to_container_fn = to_container::fn<ToContainer, Args...>;
} // namespace detail

template <template <typename...> class ContT, typename... Args, detail::to_container = detail::to_container{}>
requires(!std::ranges::range<Args> && ...) constexpr auto to(Args &&...args)
    -> detail::to_container_fn<detail::wrap<ContT>, Args...> {
    detail::to_container_fn<detail::wrap<ContT>, Args...> fn;
    fn.args = std::forward_as_tuple(std::forward<Args>(args)...);
    return fn;
}

template <template <typename...> class ContT, std::ranges::input_range Rng, typename... Args>
requires std::ranges::range<Rng>
constexpr auto to(Rng &&rng, Args &&...args) {
    return detail::to_container_fn<detail::wrap<ContT>, Args...>{}(std::forward<Rng>(rng), std::forward<Args>(args)...);
}

template <typename Cont, typename... Args, detail::to_container = detail::to_container{}>
requires(!std::ranges::range<Args> && ...) constexpr auto to(Args &&...args) -> detail::to_container_fn<Cont, Args...> {
    detail::to_container_fn<Cont, Args...> fn;
    fn.args = std::forward_as_tuple(std::forward<Args>(args)...);
    return fn;
}

template <typename Cont, std::ranges::input_range Rng, typename... Args>
requires detail::recursive_container_convertible<Cont, Rng>
constexpr auto to(Rng &&rng, Args &&...args) -> Cont {
    return detail::to_container_fn<Cont, Args...>{}(std::forward<Rng>(rng), std::forward<Args>(args)...);
}

} // namespace kas::ranges
