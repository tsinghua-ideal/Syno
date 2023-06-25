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
    const Size& size() const noexcept override { return domain; }
    std::size_t hash() const noexcept override {
        using namespace std::string_view_literals;
        constexpr int SizeTypeWidth = std::numeric_limits<std::size_t>::digits;
        constexpr int ExpectedMaximumIterators = 8;
        std::size_t h = DimensionTypeHash(DimensionType::Iterator);
        static const auto iteratorIndexHash = std::hash<std::string_view>{}("IteratorIndex"sv);
        HashCombine(h, std::rotl(iteratorIndexHash, SizeTypeWidth / ExpectedMaximumIterators * index));
        return h;
    }
    constexpr DimensionType type() const noexcept override { return DimensionType::Iterator; }
    void accept(DimVisitor& visitor) const final override;
    const Color & getColor() const override { return Color::None; }

    std::size_t getIndex() const { return index; }
    std::string getName() const {
        return "i_" + std::to_string(index);
    }
};

} // namespace kas
