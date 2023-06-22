#pragma once

#include "KAS/Core/Dimension.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

class Iterator final: public DimensionImpl {
    std::size_t index;
    Size domain;
public:
    explicit Iterator(std::size_t index, auto&& domain):
        index { index },
        domain { std::forward<decltype(domain)>(domain) }
    {}
    inline const Size& size() const noexcept override { return domain; }
    inline std::size_t hash() const noexcept override {
        std::size_t h = std::hash<DimensionType>{}(DimensionType::Iterator);
        HashCombine(h, index);
        return h;
    }
    constexpr DimensionType type() const noexcept override { return DimensionType::Iterator; }
    void accept(DimVisitor& visitor) const final override;
    const Color & getColor() const override { return Color::None; }

    inline std::size_t getIndex() const { return index; }
    inline std::string getName() const {
        return "i_" + std::to_string(index);
    }
};

} // namespace kas
