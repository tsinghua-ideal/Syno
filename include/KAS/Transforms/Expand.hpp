#pragma once

#include "KAS/Core/Expand.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class ExpandOp final: public Expand, public PrimitiveOp {
public:
    static constexpr DimensionType Type = DimensionType::Expand;

    ExpandOp(const Dimension& output):
        Expand { output },
        PrimitiveOp { Color { output.getColor() } }
    {}
    ExpandOp(const ExpandOp&) = delete;
    ExpandOp(ExpandOp&&) = delete;

    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return DimensionTypeHash(Type); }
    std::size_t opHash() const noexcept final override {
        std::size_t h = initialHash();
        HashCombineRaw(h, output.hash());
        return h;
    }
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }

    bool canApplyToInterface(const GraphHandle& interface) const final override {
        return interface.contains(output);
    }
    GraphHandle applyToInterface(const GraphHandle& interface) const final override;

    bool operator==(const ExpandOp& other) const noexcept {
        return output == other.output;
    }

    std::string description(const BindingContext& ctx) const final override {
        return fmt::format("-> {}", output.description(ctx));
    }
    std::string descendantsDescription(const BindingContext& ctx) const final override {
        return fmt::format("-> {}", output.descendantsDescription(ctx));
    }

    struct GenerateOptions {
        const BindingContext& ctx;
        bool disallowMergeInputAndWeight;
        bool disallowTile;
        std::size_t maxExpansionMultiplier;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        GenerateAttempts,
        DisallowedAttempts,
        SuccessfulGenerations,
    )
    static std::vector<const ExpandOp *> Generate(PrimitiveOpStore& store, const GraphHandle& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<ExpandOp>);

} // namespace kas
