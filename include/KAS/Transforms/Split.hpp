#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


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
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        ConflictingColors,
        CounteractedMerges,
        DisallowedDiscontinuousViews,
        UselessImmediateReductions,
        SuccessfulGenerations,
    )
    static std::vector<const SplitOp *> Generate(DimensionStore& store, const ColoredInterface& interface, const GenerateOptions& options);
};

} // namespace kas
