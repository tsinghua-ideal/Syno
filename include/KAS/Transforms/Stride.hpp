#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class StrideOp final: public RepeatLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Stride;
    class Input final: public RepeatLikeOp::Input {
    public:
        Input(const StrideOp* op):
            RepeatLikeOp::Input { op }
        {}
        const Size& size() const noexcept override { return getDerivedOp<StrideOp>()->sz; }
        constexpr DimensionType type() const noexcept override { return Type; }
    };

protected:
    Size stride;
    Size sz;
    Input input;

public:
    StrideOp(const Dimension& output, const Size& stride);
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    const Size& getStride() const { return stride; }

    bool operator==(const StrideOp& other) const noexcept {
        return output == other.output && stride == other.stride;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
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
        SuccessfulGenerations,
    )
    static std::vector<const StrideOp *> Generate(PrimitiveOpStore& store, const Dimensions& interface, const GenerateOptions& options);
};

} // namespace kas
