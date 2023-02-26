#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MergeOp final: public MergeLikePrimitiveOp {
    Size block;
    Size major;
public:
    MergeOp(auto&& output, Order order, auto&& block):
        MergeLikePrimitiveOp { std::forward<decltype(output)>(output), order },
        block { std::forward<decltype(block)>(block) },
        major { this->output.size() / this->block }
    {}

    inline const Size& size() const noexcept override {
        switch (order) {
            case Order::Left:
                return major;
            case Order::Right:
                return block;
        }
    }
    std::size_t initialHash() const noexcept override;
    constexpr DimensionType type() const noexcept override { return DimensionType::Merge; }

    IteratorValue value(const IteratorValue& output) const override;

    inline bool operator==(const MergeOp& other) const noexcept {
        return output == other.output && order == other.order && block == other.block;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<NextMergeLike> Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
