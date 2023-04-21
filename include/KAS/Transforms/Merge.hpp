#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MergeOp final: public MergeLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Merge;
    class Input final: public MergeLikeOp::Input {
    public:
        inline Input(const MergeOp* op, Order order):
            MergeLikeOp::Input { op, order }
        {}
        const Size& size() const noexcept override;
        constexpr DimensionType type() const noexcept override { return Type; }
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
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    inline Dimension getInputL() const override { return &inputLhs; }
    inline Dimension getInputR() const override { return &inputRhs; }
    Values value(const Values& known) const override;

    static std::size_t CountColorTrials;
    static std::size_t CountColorSuccesses;
    bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const override;

    inline bool operator==(const MergeOp& other) const noexcept {
        return output == other.output && minorSize == other.minorSize;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<const MergeOp *> Generate(DimensionStore& store, const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options);
};

} // namespace kas
