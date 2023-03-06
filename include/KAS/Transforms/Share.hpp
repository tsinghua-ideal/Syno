#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShareOp final: public MergeLikeOp {
public:
    class Input final: public MergeLikeOp::Input {
    public:
        inline Input(const ShareOp* op, Order order):
            MergeLikeOp::Input { op, order }
        {}
        inline const Size& size() const noexcept override { return op->output.size(); }
        std::size_t hash() const noexcept override;
        constexpr DimensionType type() const noexcept override { return DimensionType::Share; }
    };

protected:
    Input inputLhs, inputRhs;

public:
    // Only std::make_unique allowed!
    ShareOp(auto&& output):
        MergeLikeOp { std::forward<decltype(output)>(output) },
        inputLhs { this, Order::Left },
        inputRhs { this, Order::Right }
    {}
    inline std::pair<Dimension, Dimension> getInputs() const override { return { &inputLhs, &inputRhs }; }
    std::pair<IteratorValue, IteratorValue> value(const IteratorValue& output) const override;

    inline bool operator==(const ShareOp& other) const noexcept {
        return output == other.output;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<std::unique_ptr<ShareOp>> Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
