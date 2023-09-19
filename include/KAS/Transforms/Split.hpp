#pragma once

#include "KAS/Core/DimVisitor.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

struct ReshapeBlockNeighbors {
    const MergeOp *left = nullptr;
    const MergeOp *right = nullptr;
    using Self = ReshapeBlockNeighbors;
    auto separatedBy(const MergeOp *separator) const -> std::pair<Self, Self>;
    auto isAdjacentTo(const Self& rhs) const -> bool;
    auto combinedWith(const Self& rhs) const -> Self;
};

// Canonicalize reshape.
// First we only allow Split's above Merge's,
// then we check for redundant Split's.
// The rule is simple. After the sequence of Merge's, we obtain the smallest reshape blocks,
// and if the blocks that are adjacent get combined by Split's again, this is illegal.
struct ReshapeCanonicalizer: public BottomTopDimVisitor<ReshapeCanonicalizer, ReshapeBlockNeighbors> {
    using Adjacent = ReshapeBlockNeighbors;
    auto transform(const Iterator&) const -> Adjacent;
    auto transform(const Reduce&) const -> Adjacent;
    auto transform(const RepeatLikeOp::Input&) const -> Adjacent;
    auto transform(const SplitLikeOp::Input& dim) const -> Adjacent;
    auto transform(const MergeLikeOp::Input& dim) const -> std::pair<Adjacent, Adjacent>;
};

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
    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return DimensionTypeHash(Type); }
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }
    Dimension getInput() const override { return &input; }
    Values value(const Values& known) const override;

    bool operator==(const SplitOp& other) const noexcept {
        return outputLhs == other.outputLhs && outputRhs == other.outputRhs;
    }

    struct GenerateOptions {
        const Graph& graph;
        bool disallowDiscontinuousView;
        bool disallowSplitLAboveUnfold;
        bool disallowSplitRAboveUnfold;
        bool disallowSplitRAboveStride;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        CounteractedMerges,
        DisallowedDiscontinuousViews,
        UselessImmediateReductions,
        SuccessfulGenerations,
    )
    static std::vector<const SplitOp *> Generate(PrimitiveOpStore& store, const GraphHandle& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<SplitOp>);

} // namespace kas
