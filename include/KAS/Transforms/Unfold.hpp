#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class UnfoldOp final: public SplitLikeOp {
    bool isEqual(const Operation& other) const override;
public:
    static constexpr DimensionType Type = DimensionType::Unfold;
    class Input final: public SplitLikeOp::Input {
    public:
        Input(const UnfoldOp* op):
            SplitLikeOp::Input { op }
        {}
        const Size& size() const override { return getDerivedOp<UnfoldOp>()->outputLhs.size(); }
        constexpr DimensionType type() const noexcept override { return Type; }
        Color computeColor(const GraphBuilder& graphBuilder) const override;
    };

protected:
    Input input;

public:
    UnfoldOp(const Dimension& outputLhs, const Dimension& outputRhs);
    const Size& getWindow() const { return outputRhs.size(); }
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return DimensionTypeHash(Type); }
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    struct GenerateOptions {
        const BindingContext& ctx;
        const Graph& graph;
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
        DoubleReduction,
        EvenKernelSize,
        KernelAbsolutelyTooLarge,
        KernelRelativelyTooLarge,
        CanonicalizedUnfoldChains,
        SuccessfulGenerations,
    )
    static std::vector<const UnfoldOp *> Generate(OperationStore& store, const Topmost& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<UnfoldOp>);

} // namespace kas
