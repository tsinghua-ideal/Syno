#pragma once

#include "KAS/Core/Expand.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class ExpandOp final: public Expand, public PrimitiveOp {
public:
    static constexpr DimensionType Type = DimensionType::Expand;

    ExpandOp(const Dimension& output):
        Expand { output }
    {}
    ExpandOp(const ExpandOp&) = delete;
    ExpandOp(ExpandOp&&) = delete;

    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override;
    std::size_t opHash() const noexcept final override;
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }

    static const ExpandOp *FromRaw(const Expand *raw) { return &dynamic_cast<const ExpandOp&>(*raw); }

    bool canApplyToInterface(const GraphHandle& interface) const final override;
    void applyToInterface(GraphHandle& interface) const final override;

    bool operator==(const ExpandOp& other) const noexcept;

    std::string description(const BindingContext& ctx) const final override;
    std::string descendantsDescription(const BindingContext& ctx) const final override;

    struct Usage {
        Graph::DimensionSet sharedWeightDims;
        Size mergedInputAndWeight;
    };
    static Usage GetUsage(const BindingContext& ctx, const Graph& graph);

    struct GenerateOptions {
        const BindingContext& ctx;
        bool disallowMergeInputAndWeight;
        bool disallowTile;
        bool disallowShareWeights;
        std::size_t maxExpansionRepeatMultiplier;
        std::size_t maxExpansionMergeMultiplier;
        std::size_t maxExpansionWeightsSharingDimSize;
        std::size_t minExpansionWeightsSharingDimSize;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        SuccessfulGenerations,
    )
    static std::vector<const ExpandOp *> Generate(PrimitiveOpStore& store, const Topmost& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<ExpandOp>);

// Helper class.
class RepeatOp {
public:
    enum Kind: bool {
        Repeat, // torch.repeat_interleave
        Tile, // torch.tile
    };
private:
    const ExpandOp& expandOp;
    const MergeOp& mergeOp;
    Dimension input;
    Kind kind;
public:
    RepeatOp(const ExpandOp& expandOp, const MergeOp& mergeOp);
    const Dimension& output;
    Kind getKind() const { return kind; }
    const Size& getMultiplier() const { return expandOp.output.size(); }
    const Dimension& getInput() const { return input; }
    std::string description(const BindingContext& ctx) const;
};

} // namespace kas
