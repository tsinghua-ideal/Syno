#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


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
    MergeOp(const Dimension& output, const Size& block);
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInputL() const override { return &inputLhs; }
    Dimension getInputR() const override { return &inputRhs; }
    Values value(const Values& known) const override;

    bool operator==(const MergeOp& other) const noexcept {
        return output == other.output && minorSize == other.minorSize;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        float minimumRatio = 2.0f;
        bool disallowMergeWithLargeBlockAboveStride;
        // This canonicalization deviates a lot from original semantics. Enable with caution!
        bool disallowMergeWithLargeBlockAboveUnfold;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        ConteractedSplits,
        UselessImmediateReductions,
        BlockRelativelyTooLarge,
        DisallowedAboveStride,
        DisallowedAboveUnfold,
        SuccessfulGenerations,
    )
    static std::vector<const MergeOp *> Generate(PrimitiveOpStore& store, const Dimensions& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<MergeOp>);

} // namespace kas
