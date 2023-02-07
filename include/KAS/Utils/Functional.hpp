#pragma once

#include "KAS/Utils/Tuple.hpp"
#include <cstddef>
#include <tuple>
#include <type_traits>


namespace kas {

template<std::size_t N>
auto ReverseArguments(auto&& f) -> decltype(auto) {
    return [&]<typename... Args>(Args&&... args) requires(sizeof...(Args) == N) {
        return std::apply(f, ReverseTuple(std::forward_as_tuple(std::forward<Args>(args)...)));
    };
}

} // namespace kas
