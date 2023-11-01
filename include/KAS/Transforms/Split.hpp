#pragma once

#include "KAS/Core/DimVisitor.hpp"
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
        const Size& size() const override { return getDerivedOp<SplitOp>()->sz; }
        constexpr DimensionType type() const noexcept override { return Type; }
    };

protected:
    Size sz;
    Input input;

public:
    SplitOp(const Dimension& outputLhs, const Dimension& outputRhs);
    const Size& getBlock() const { return outputRhs.size(); }
    const Size& getGroups() const { return outputLhs.size(); }
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return DimensionTypeHash(Type); }
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    bool operator==(const SplitOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        const Graph& graph;
        bool disallowSplitLAboveUnfold;
        bool disallowSplitRAboveUnfold;
        bool disallowSplitRAboveStride;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        CounteractedMergesAndReduces,
        CounteractedUnorderedMerges,
        InvalidProductSize,
        SuccessfulGenerations,
    )
    static std::vector<const SplitOp *> Generate(PrimitiveOpStore& store, const Topmost& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<SplitOp>);

} // namespace kas
