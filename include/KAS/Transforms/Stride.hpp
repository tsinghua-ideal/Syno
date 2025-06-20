#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class StrideOp final: public RepeatLikeOp {
    bool isEqual(const Operation& other) const override;
public:
    static constexpr DimensionType Type = DimensionType::Stride;
    class Input final: public RepeatLikeOp::Input {
    public:
        Input(const StrideOp* op):
            RepeatLikeOp::Input { op }
        {}
        const Size& size() const override { return getDerivedOp<StrideOp>()->sz; }
        constexpr DimensionType type() const noexcept override { return Type; }
        Color computeColor(const GraphBuilder &graphBuilder) const override;
    };

protected:
    Size stride;
    Size sz;
    Input input;

public:
    StrideOp(const Dimension& output, const Size& stride);
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    const Size& getStride() const { return stride; }

    struct GenerateOptions {
        const BindingContext& ctx;
        const Allowance& allowance;
        // stride * outputDim.size() == inputDim.size() <= maxStridedDimSize. This should correspond to UnfoldOp::GenerateOptions::maxUnfoldKernelSize.
        std::size_t maxStridedDimSize = 30;
        bool disallowStrideAboveSplit;
        bool disallowStrideAboveMergeR;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        SizeTooLarge,
        InvalidProductSize,
        SuccessfulGenerations,
    )
    static std::vector<const StrideOp *> Generate(OperationStore& store, const Topmost& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<StrideOp>);

} // namespace kas
