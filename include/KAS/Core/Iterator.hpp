#pragma once

#include "KAS/Core/Dimension.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

class Iterator final: public DimensionImpl {
    std::size_t index;
    Size domain;
    bool isUnordered;
public:
    explicit Iterator(std::size_t index, const Size& domain, bool isUnordered = false):
        index { index },
        domain { std::forward<decltype(domain)>(domain) },
        isUnordered { isUnordered }
    {}
    const Size& size() const override { return domain; }
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
    const PrimitiveOp *getOpBelow() const override { return nullptr; }
    Color computeColor(const GraphBuilder& graphBuilder) const override { return isUnordered ? Color().setUnordered(this) : Color(); }

    std::size_t getIndex() const { return index; }
    std::string getName() const {
        return "i_" + std::to_string(index);
    }
};

} // namespace kas
