#pragma once

#include "KAS/Core/Dimension.hpp"

namespace kas {

class Iterator final: public DimensionImpl {
    std::size_t index;
    Size domain;
public:
    explicit inline Iterator(std::size_t index, auto&& domain):
        index { index },
        domain { std::forward<decltype(domain)>(domain) }
    {}
    inline const Size& size() const noexcept override { return domain; }
    inline std::size_t initialHash() const noexcept override { return index; }
    constexpr DimensionType type() const noexcept override { return DimensionType::Iterator; }
};

} // namespace kas
