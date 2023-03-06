#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MergeOp final: public MergeLikeOp {
public:
    class Input final: public MergeLikeOp::Input {
    public:
        inline Input(const MergeOp* op, Order order):
            MergeLikeOp::Input { op, order }
        {}
        const Size& size() const noexcept override;
        std::size_t hash() const noexcept override;
        constexpr DimensionType type() const noexcept override { return DimensionType::Merge; }
    };

protected:
    Size minorSize;
    Size majorSize;
    Input inputLhs, inputRhs;

public:
    MergeOp(auto&& output, auto&& block):
        MergeLikeOp { std::forward<decltype(output)>(output) },
        minorSize { std::forward<decltype(block)>(block) },
        majorSize { this->output.size() / this->minorSize },
        inputLhs { this, Order::Left },
        inputRhs { this, Order::Right }
    {}
    inline std::pair<Dimension, Dimension> getInputs() const override { return { &inputLhs, &inputRhs }; }
    std::pair<IteratorValue, IteratorValue> value(const IteratorValue& output) const override;

    inline bool operator==(const MergeOp& other) const noexcept {
        return output == other.output && minorSize == other.minorSize;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<std::unique_ptr<MergeOp>> Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
