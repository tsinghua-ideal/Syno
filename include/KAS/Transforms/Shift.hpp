#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShiftOp final: public RepeatLikePrimitiveOp {
    int shift;
public:
    ShiftOp(auto&& output, int shift):
        RepeatLikePrimitiveOp { std::forward<decltype(output)>(output) },
        shift { shift }
    {}
    inline const Size& size() const noexcept override { return output.size(); }
    std::size_t initialHash() const noexcept override;
    constexpr DimensionType type() const noexcept override { return DimensionType::Shift; }

    IteratorValue value(const IteratorValue& output) const override;

    inline bool operator==(const ShiftOp& other) const noexcept {
        return output == other.output && shift == other.shift;
    }

    static std::vector<NextRepeatLike> Generate(DimensionStore& store, const Interface& outputShape);
};

} // namespace kas
