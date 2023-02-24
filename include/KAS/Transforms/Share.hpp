#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShareOp final: public MergeLikePrimitiveOp {
public:
    using MergeLikePrimitiveOp::MergeLikePrimitiveOp;

    inline const Size& size() const noexcept override { return output.size(); }
    // Since ShareOp keeps no metadata, the initial hash is the same for all ShareOps.
    constexpr std::size_t initialHash() const noexcept override { return static_cast<std::size_t>(type()); }
    constexpr DimensionType type() const noexcept override { return DimensionType::Share; }

    IteratorValue value(const IteratorValue& output) const override;

    inline bool operator==(const ShareOp& other) const noexcept {
        return output == other.output && order == other.order;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<std::pair<Dimension, Dimension>> Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
