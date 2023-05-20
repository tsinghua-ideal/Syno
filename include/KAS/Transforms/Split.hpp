#pragma once

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class SplitOp final: public SplitLikeOp {
public:
    static constexpr DimensionType Type = DimensionType::Split;
    class Input final: public SplitLikeOp::Input {
    public:
        Input(const SplitOp* op):
            SplitLikeOp::Input { op }
        {}
        const Size& size() const noexcept override { return getDerivedOp<SplitOp>()->sz; }
        constexpr DimensionType type() const noexcept override { return Type; }
    };

protected:
    Size sz;
    Input input;

public:
    SplitOp(auto&& outputLhs, auto&& outputRhs):
        SplitLikeOp { std::forward<decltype(outputLhs)>(outputLhs), std::forward<decltype(outputRhs)>(outputRhs) },
        sz { this->outputLhs.size() * this->outputRhs.size() },
        input { this }
    {}
    constexpr DimensionType getType() const noexcept override { return Type; }
    constexpr std::size_t initialHash() const noexcept override { return static_cast<std::size_t>(Type); }
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    bool operator==(const SplitOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        bool disallowDiscontinuousView;
        bool disallowSplitRAboveUnfold;
        bool disallowSplitRAboveStride;
    };
    static inline std::size_t CountGenerateInvocations = 0;
    static inline std::size_t CountGenerateAttempts = 0; // Equals the sum of below.
    static inline std::size_t CountDisallowedAttempts = 0;
    static inline std::size_t CountConflictingColors = 0;
    static inline std::size_t CountCounteractedMerges = 0;
    static inline std::size_t CountDisallowedDiscontinuousViews = 0;
    static inline std::size_t CountUselessImmediateReductions = 0;
    static inline std::size_t CountSuccessfulGenerations = 0;
    static std::vector<const SplitOp *> Generate(DimensionStore& store, const ColoredInterface& interface, GenerateOptions options);
};

} // namespace kas
