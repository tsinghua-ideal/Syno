#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class UnfoldOp final: public SplitLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Unfold;
    class Input final: public SplitLikeOp::Input {
    public:
        Input(const UnfoldOp* op):
            SplitLikeOp::Input { op }
        {}
        const Size& size() const noexcept override { return getDerivedOp<UnfoldOp>()->outputLhs.size(); }
        constexpr DimensionType type() const noexcept override { return Type; }
    };

protected:
    Input input;

public:
    UnfoldOp(const Dimension& outputLhs, const Dimension& outputRhs);
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return DimensionTypeHash(Type); }
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    bool operator==(const UnfoldOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        float minimumRatio = 2.0f;
        // kernel.size() <= maxUnfoldKernelSize. This should correspond to StrideOp::GenerateOptions::maxStridedDimSize.
        std::size_t maxUnfoldKernelSize = 30;
        bool requiresOddKernelSizeInUnfold = false;
        bool disallowUnfoldLAboveSplit;
        bool canonicalizeUnfoldOrder;
        bool disallowUnfoldLAboveShift;
        // This canonicalization deviates a lot from original semantics. Enable with caution!
        bool disallowUnfoldLAboveMergeR;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        KernelAbsolutelyTooLarge,
        KernelRelativelyTooLarge,
        CanonicalizedUnfoldChains,
        SuccessfulGenerations,
    )
    static std::vector<const UnfoldOp *> Generate(PrimitiveOpStore& store, const Dimensions& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<UnfoldOp>);

} // namespace kas
