#pragma once

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class ExpandOp final: public PrimitiveOp {
public:
    static constexpr DimensionType Type = DimensionType::Expand;

    Dimension output;

    ExpandOp(const Dimension& output):
        PrimitiveOp { Color { output.getColor() } },
        output { output }
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

    bool canApplyToInterface(const Dimensions& interface) const final override {
        return interface.contains(output);
    }
    Dimensions applyToInterface(const Dimensions& interface) const final override;

    std::string description(const BindingContext& ctx) const final override {
        return fmt::format("-> {}", output.description(ctx));
    }
    std::string descendantsDescription(const BindingContext& ctx) const final override {
        return fmt::format("-> {}", output.descendantsDescription(ctx));
    }

    struct GenerateOptions {
        bool disallowMergeInputAndWeight;
    };
    static std::vector<const ExpandOp *> Generate(PrimitiveOpStore& store, const Dimensions& interface, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<ExpandOp>);

} // namespace kas
