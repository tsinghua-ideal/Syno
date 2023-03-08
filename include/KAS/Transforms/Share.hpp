#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShareOp final: public MergeLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Share;
    class Input final: public MergeLikeOp::Input {
    public:
        inline Input(const ShareOp* op, Order order):
            MergeLikeOp::Input { op, order }
        {}
        inline const Size& size() const noexcept override { return op->output.size(); }
        constexpr DimensionType type() const noexcept override { return Type; }
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
    constexpr DimensionType getType() const noexcept override { return Type; }
    constexpr std::size_t initialHash() const noexcept override { return static_cast<std::size_t>(Type); }
    inline std::pair<Dimension, Dimension> getInputs() const override { return { &inputLhs, &inputRhs }; }
    std::pair<IteratorValue, IteratorValue> value(const IteratorValue& output) const override;
    bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const override;

    inline bool operator==(const ShareOp& other) const noexcept {
        return output == other.output;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<const ShareOp *> Generate(DimensionStore& store, const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options);
};

} // namespace kas
