#pragma once

#include "KAS/Core/Expand.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class ExpandOp final: public Expand, public PrimitiveOp {
    bool isEqual(const Operation& other) const override;
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

    std::string description(const BindingContext& ctx) const final override;
    std::string descendantsDescription(const BindingContext& ctx) const final override;

    struct GenerateOptions {
        const BindingContext& ctx;
        bool disallowTile;
        std::size_t maxExpansionRepeatMultiplier;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        SuccessfulGenerations,
    )
    static std::vector<const ExpandOp *> Generate(OperationStore& store, const Topmost& interface, const GenerateOptions& options);
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
