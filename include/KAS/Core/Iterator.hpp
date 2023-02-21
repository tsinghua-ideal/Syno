#pragma once

#include <boost/container_hash/hash_fwd.hpp>

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
    inline std::size_t initialHash() const noexcept override {
        auto h = index;
        boost::hash_combine(h, "Iterator");
        return h;
    }
    constexpr DimensionType type() const noexcept override { return DimensionType::Iterator; }
};

} // namespace kas
