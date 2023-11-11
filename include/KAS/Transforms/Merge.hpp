#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class MergeOp final: public MergeLikeOp {
    bool isEqual(const Operation& other) const override;
public:
    static constexpr DimensionType Type = DimensionType::Merge;
    class Input final: public MergeLikeOp::Input {
    public:
        Input(const MergeOp* op, Order order):
            MergeLikeOp::Input { op, order }
        {}
        const Size& size() const override;
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
    const Size& getBlock() const { return minorSize; }
    const Size& getGroup() const { return majorSize; }
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInputL() const override { return &inputLhs; }
    Dimension getInputR() const override { return &inputRhs; }
    Values value(const Values& known) const override;

    struct GenerateOptions {
        const BindingContext& ctx;
        const Graph& graph;
        const Allowance& allowance;
        bool disallowMergeWithLargeBlockAboveStride;
        // This canonicalization deviates a lot from original semantics. Enable with caution!
        bool disallowMergeWithLargeBlockAboveUnfold;
        float maximumValidReshapeShiftPattern;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        ExceedsMaxValidReshapeShiftPattern,
        UselessImmediateReductions,
        BlockRelativelyTooLarge,
        DisallowedAboveStride,
        DisallowedAboveUnfold,
        UnorderedSizeOrderingViolated,
        SuccessfulGenerations,
    )
    static std::vector<const MergeOp *> Generate(OperationStore& store, const Topmost& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<MergeOp>);

} // namespace kas
