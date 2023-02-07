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

} // namespace kas
