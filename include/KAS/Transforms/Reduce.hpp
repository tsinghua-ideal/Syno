#pragma once

#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ReduceOp final: public Reduce, public PrimitiveOp {
public:
    static constexpr DimensionType Type = DimensionType::Reduce;

    ReduceOp(std::size_t priority, const Size& domain, MapType mapType, ReduceType reduceType):
        Reduce { priority, domain, mapType, reduceType },
        PrimitiveOp { Color::None }
    {}
    ReduceOp(const Reduce&) = delete;
    ReduceOp(Reduce&&) = delete;

    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return hash(); }
    std::size_t opHash() const noexcept override { return initialHash(); }
    using Reduce::accept;
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }

    const Reduce *getRaw() const { return this; }
    Dimension getInput() const { return this; }

    bool canApplyToInterface(const GraphHandle& interface) const override;
    GraphHandle applyToInterface(const GraphHandle& interface) const override;

    bool operator==(const ReduceOp& other) const noexcept {
        return getMap() == other.getMap() && getReduce() == other.getReduce() && getPriority() == other.getPriority() && size() == other.size();
    }

    std::string description(const BindingContext& ctx) const override;
    std::string descendantsDescription(const BindingContext& ctx) const override;

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
        Size outputSize;
        std::size_t maxFLOPs;
        std::size_t maximumReductions;
    };
    static std::vector<const ReduceOp *> Generate(PrimitiveOpStore& store, const std::vector<const ReduceOp *>& current, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<ReduceOp>);

} // namespace kas
