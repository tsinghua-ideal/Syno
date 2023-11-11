#pragma once

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

class ReduceBase {
public:
    static constexpr DimensionType Type = DimensionType::Reduce;
    static constexpr std::size_t ExpectedMaximumReduces = 6;

    enum class ReduceType {
        Sum,
        Max,
        Mean,
        Min,
        Product,
        ReduceTypeCount
    };
    static std::string what(ReduceType);

protected:
    // Do not allow Map's, and assume that all reductions are commutative and associative.
    // Then we do not need to assign a priority for this reduction.
    // This aids search.
    // std::size_t priority;
    Size domain;

    ReduceType reduceType;

    ReduceBase(const Size& domain, ReduceType reduceType):
        domain { domain },
        reduceType { reduceType }
    {}

public:
    const Size& getDomain() const { return domain; }
    ReduceType getReduce() const { return reduceType; }
    std::string whatReduce() const;

    bool equalsTo(const ReduceBase& other) const noexcept;
    // Excludes multiplicity.
    std::size_t pureHash() const noexcept;

    virtual ~ReduceBase() = default;
};

class Reduce final: public DimensionImpl {
    friend class ReduceOp;

    const ReduceBase& base;

    // But still we need to identify each reduction, even they have the same domains.
    // So we need to somehow number them.
    // If this is the first reduce of this domain, multiplicity == 0. If this is the second reduce of this domain, multiplicity == 1, and so on.
    std::size_t multiplicity;

    Reduce(const ReduceBase& base, std::size_t multiplicity):
        base { base },
        multiplicity { multiplicity }
    {}

public:
    static constexpr std::size_t ExpectedMaximumReduces = ReduceBase::ExpectedMaximumReduces;
    using ReduceType = ReduceBase::ReduceType;

    const Size& size() const override { return base.getDomain(); }
    bool operator==(const Reduce& other) const noexcept {
        return base.equalsTo(other.base) && multiplicity == other.multiplicity;
    }
    // Includes multiplicity.
    std::size_t hash() const noexcept override;
    constexpr DimensionType type() const noexcept override { return ReduceBase::Type; }
    void accept(DimVisitor& visitor) const override;
    const PrimitiveOp *getOpBelow() const override { return nullptr; }
    Color computeColor(const GraphBuilder& graphBuilder) const override {
        return Color().setUnordered(this).setHeight(1).setEndsUpReduce(true);
    }

    const ReduceBase& getBase() const { return base; }
    ReduceType getReduce() const { return base.getReduce(); }
    std::string whatReduce() const { return base.whatReduce(); }

    // Dictionary order: reduceType, domain, multiplicity.
    static std::strong_ordering LexicographicalCompare(const Reduce& a, const Reduce& b) noexcept {
        auto reduceType = a.getReduce() <=> b.getReduce();
        if (reduceType != 0) return reduceType;
        auto domain = Size::LexicographicalCompare(a.size(), b.size());
        if (domain != 0) return domain;
        return a.getMultiplicity() <=> b.getMultiplicity();
    }
    static bool LexicographicalLessThan(const Reduce& a, const Reduce& b) noexcept {
        return LexicographicalCompare(a, b) == std::strong_ordering::less;
    }

    std::size_t getMultiplicity() const { return multiplicity; }

    std::string description(const BindingContext& ctx) const;

    // FOR DEBUG USAGE ONLY!
    std::string debugDescription() const;
};

using ReductionShapeView = AbstractShape<const std::vector<const Reduce *>&, [](const Reduce *r) -> const Size& { return r->size(); }>;

} // namespace kas
