#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


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
    UnfoldOp(auto&& outputLhs, auto&& outputRhs):
        SplitLikeOp { std::forward<decltype(outputLhs)>(outputLhs), std::forward<decltype(outputRhs)>(outputRhs) },
        input { this }
    {}
    constexpr DimensionType getType() const noexcept override { return Type; }
    constexpr std::size_t initialHash() const noexcept override { return static_cast<std::size_t>(Type); }
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    // Absorb dataDiscardingFlag in outputRhs.
    ColoredInterface applyToInterface(const ColoredInterface& interface) const override;

    bool operator==(const UnfoldOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimLowerBound;
        float minimumRatio = 2.0f;
        // kernel.size() <= maxUnfoldKernelSize. This should correspond to StrideOp::GenerateOptions::maxStridedDimSize.
        std::size_t maxUnfoldKernelSize = 30;
        bool disallowUnfoldLAboveSplit;
        bool canonicalizeUnfoldOrder;
        bool disallowUnfoldLAboveShift;
        // This canonicalization deviates a lot from original semantics. Enable with caution!
        bool disallowUnfoldLAboveMergeR;
    };
    static inline std::size_t CountGenerateInvocations = 0;
    static inline std::size_t CountGenerateAttempts = 0; // Equals the sum of below.
    static inline std::size_t CountDisallowedAttempts = 0;
    static inline std::size_t CountConflictingColors = 0;
    static inline std::size_t CountKernelAbsolutelyTooLarge = 0;
    static inline std::size_t CountKernelRelativelyTooLarge = 0;
    static inline std::size_t CountCanonicalizedUnfoldChains = 0;
    static inline std::size_t CountSuccessfulGenerations = 0;
    static std::vector<const UnfoldOp *> Generate(DimensionStore& store, const ColoredInterface& interface, GenerateOptions options);
};

} // namespace kas
