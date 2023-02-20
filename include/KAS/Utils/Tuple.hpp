#pragma once

#include <tuple>


namespace kas {

template<typename TupR, typename Tup = std::remove_reference_t<TupR>,
         auto N = std::tuple_size_v<Tup>>
constexpr auto ReverseTuple(TupR&& t) {
    return [&t]<auto... I>(std::index_sequence<I...>) {
        constexpr std::array is{(N - 1 - I)...};
        return std::tuple<std::tuple_element_t<is[I], Tup>...>{
            std::get<is[I]>(std::forward<TupR>(t))...
        };
    }(std::make_index_sequence<N>{});
}

template <typename Tuple, typename F, std::size_t ...Indices>
void TupleForEachImpl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>) {
    using swallow = int[];
    (void)swallow { 1, (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})... };
}

template <typename Tuple, typename F>
void TupleForEach(Tuple&& tuple, F&& f) {
    constexpr std::size_t N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
    TupleForEachImpl(std::forward<Tuple>(tuple), std::forward<F>(f), std::make_index_sequence<N>{});
}

template <typename T, typename Tuple>
struct TupleHasType;

template <typename T>
struct TupleHasType<T, std::tuple<>> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct TupleHasType<T, std::tuple<U, Ts...>> : TupleHasType<T, std::tuple<Ts...>> {};

template <typename T, typename... Ts>
struct TupleHasType<T, std::tuple<T, Ts...>> : std::true_type {};

template <typename T, typename... Ts>
constexpr bool TupleHasTypeV = TupleHasType<T, std::tuple<Ts...>>::value;

} // namespace kas
