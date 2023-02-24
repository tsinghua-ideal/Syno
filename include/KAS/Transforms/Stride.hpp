#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class StrideOp final: public RepeatLikePrimitiveOp {
    Size stride;
    Size sz;
public:
    StrideOp(auto&& output, auto&& stride):
        RepeatLikePrimitiveOp { std::forward<decltype(output)>(output) },
        stride { stride },
        sz { this->output.size() * this->stride }
    {}
    inline const Size& size() const noexcept override { return sz; }
    std::size_t initialHash() const noexcept override;
    constexpr DimensionType type() const noexcept override { return DimensionType::Stride; }

    IteratorValue value(const IteratorValue& output) const override;

    inline bool operator==(const StrideOp& other) const noexcept {
        return output == other.output && stride == other.stride;
    }

    static std::vector<Dimension> Generate(DimensionStore& store, const Interface& outputShape);
};

} // namespace kas
