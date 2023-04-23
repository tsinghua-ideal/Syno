#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "KAS/Utils/Tuple.hpp"


namespace kas {

namespace detail {

struct AnyArgument {
    template<typename T>
    operator T&&() const;
};

template<typename Lambda, typename Is, typename = void>
struct CanAcceptImpl: std::false_type {};

template<typename Lambda, std::size_t ...Is>
struct CanAcceptImpl<
    Lambda,
    std::index_sequence<Is...>, 
    decltype(std::declval<Lambda>()(((void)Is, AnyArgument{})...), void())
>: std::true_type {};

template<typename Lambda, std::size_t N>
struct CanAccept: CanAcceptImpl<Lambda, std::make_index_sequence<N>> {};

template<typename Lambda, std::size_t Max, std::size_t N, typename = void>
struct LambdaDetailsImpl: LambdaDetailsImpl<Lambda, Max, N - 1> {};

template<typename Lambda, std::size_t Max, std::size_t N>
struct LambdaDetailsImpl<Lambda, Max, N, std::enable_if_t<CanAccept<Lambda, N>::value>>
{
    static constexpr bool IsVariadic = (N == Max);
    static constexpr std::size_t ArgumentCount = N;
};

template<typename Lambda, std::size_t Max = 50>
struct LambdaDetails: LambdaDetailsImpl<Lambda, Max, Max> {};

} // namespace detail

template<typename F>
constexpr std::size_t GetArgumentCount()
{
    using extractor = detail::LambdaDetails<F>;
    static_assert(!extractor::IsVariadic, "Cannot get argument count of variadic lambda");
    return extractor::ArgumentCount;
}

template<std::size_t N>
auto ReverseNArguments(auto&& f) -> decltype(auto) {
    return [&]<typename... Args>(Args&&... args) requires(sizeof...(Args) == N) {
        return std::apply(f, ReverseTuple(std::forward_as_tuple(std::forward<Args>(args)...)));
    };
}

auto ReverseArguments(auto&& f) -> decltype(auto) {
    constexpr std::size_t N = GetArgumentCount<decltype(f)>();
    if constexpr (N <= 1) {
        return std::forward<decltype(f)>(f);
    } else {
        return [&]<typename... Args>(Args&&... args) requires(sizeof...(Args) == N) {
            return std::apply(f, ReverseTuple(std::forward_as_tuple(std::forward<Args>(args)...)));
        };
    }
}

} // namespace kas
