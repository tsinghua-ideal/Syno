#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MergeOp final: public MergeLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Merge;
    class Input final: public MergeLikeOp::Input {
    public:
        Input(const MergeOp* op, Order order):
            MergeLikeOp::Input { op, order }
        {}
        const Size& size() const noexcept override;
        constexpr DimensionType type() const noexcept override { return Type; }
        bool is(DimensionTypeWithOrder ty) const noexcept override {
            return (ty == DimensionTypeWithOrder::MergeL && order == Order::Left)
                || (ty == DimensionTypeWithOrder::MergeR && order == Order::Right);
        }
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
    Dimension getInputL() const override { return &inputLhs; }
    Dimension getInputR() const override { return &inputRhs; }
    Values value(const Values& known) const override;

    bool operator==(const MergeOp& other) const noexcept {
        return output == other.output && minorSize == other.minorSize;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
        float minimumRatio = 2.0f;
        bool disallowMergeWithLargeBlockAboveStride;
        bool disallowMergeWithLargeBlockAboveUnfold;
    };
    static inline std::size_t CountGenerateInvocations = 0;
    static inline std::size_t CountGenerateAttempts = 0; // Equals the sum of below.
    static inline std::size_t CountDisallowedAttempts = 0;
    static inline std::size_t CountConteractedSplits = 0;
    static inline std::size_t CountUselessImmediateReductions = 0;
    static inline std::size_t CountBlockRelativelyTooLarge = 0;
    static inline std::size_t CountDisallowedAboveStride = 0;
    static inline std::size_t CountDisallowedAboveUnfold = 0;
    static inline std::size_t CountSuccessfulGenerations = 0;
    static std::vector<const MergeOp *> Generate(DimensionStore& store, const ColoredInterface& interface, GenerateOptions options);
};

} // namespace kas
