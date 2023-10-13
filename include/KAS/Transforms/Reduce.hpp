#pragma once

#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ReduceOp final: public ReduceBase, public PrimitiveOp {
    // Store the Reduce's with different multiplicity in an array.
    // So we can easily access them.
    std::array<Reduce, ExpectedMaximumReduces> reduces;

public:
    ReduceOp(const Size& domain, ReduceType reduceType):
        ReduceBase { domain, reduceType },
        PrimitiveOp { Color::None },
        // Store the Reduce's in the Op.
        reduces([this]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::array<Reduce, ExpectedMaximumReduces> { Reduce(*this, Is)... };
        }(std::make_index_sequence<ExpectedMaximumReduces>{}))
    {}
    ReduceOp(const Reduce&) = delete;
    ReduceOp(Reduce&&) = delete;

    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return pureHash(); }
    std::size_t opHash() const noexcept override { return initialHash(); }
    void accept(OpVisitor& visitor) const override { visitor.visit(*this); }

    const Reduce *getRaw(std::size_t multiplicity) const { return &reduces[multiplicity]; }
    static const ReduceOp *FromRaw(const Reduce *raw) { return &dynamic_cast<const ReduceOp&>(raw->getBase()); }
    Dimension getInput(std::size_t multiplicity) const { return getRaw(multiplicity); }

    std::size_t getMultiplicity(const std::vector<Dimension>& interface) const;
    // Count how many Reduce's with the same size there are.
    std::size_t getMultiplicity(const GraphHandle& interface) const;

    bool canApplyToInterface(const GraphHandle& interface) const override;
    GraphHandle applyToInterface(const GraphHandle& interface) const override;

    bool operator==(const ReduceOp& other) const noexcept {
        return ReduceBase::operator==(other);
    }

    std::string description(const BindingContext& ctx) const override;
    std::string descendantsDescription(const BindingContext& ctx) const override;

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
        Size outputSize;
        Size maxRDomSizeBase;
        std::size_t maxRDomSizeMultiplier;
        std::size_t maximumReductions;
    };
    static std::vector<const ReduceOp *> Generate(PrimitiveOpStore& store, const std::vector<const Reduce *>& current, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<ReduceOp>);

} // namespace kas
