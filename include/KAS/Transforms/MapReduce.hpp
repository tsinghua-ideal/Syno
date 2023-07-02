#pragma once

#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MapReduceOp final: public MapReduce, public PrimitiveOp {
public:
    static constexpr DimensionType Type = DimensionType::MapReduce;

public:
    MapReduceOp(std::size_t priority, const Size& domain, MapType mapType, ReduceType reduceType):
        MapReduce { priority, domain, mapType, reduceType },
        PrimitiveOp { Color::None }
    {}
    MapReduceOp(const MapReduce&) = delete;
    MapReduceOp(MapReduce&&) = delete;

    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return hash(); }
    std::size_t opHash() const noexcept override { return initialHash(); }

    const MapReduce *getRaw() const { return this; }
    Dimension getInput() const { return this; }

    bool canApplyToInterface(const Dimensions& interface) const override;
    Dimensions applyToInterface(const Dimensions& interface) const override;

    bool operator==(const MapReduceOp& other) const noexcept {
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
    static std::vector<const MapReduceOp *> Generate(PrimitiveOpStore& store, const std::vector<const MapReduceOp *>& current, const GenerateOptions& options);
};

static_assert(PrimitiveOpImpl<MapReduceOp>);

} // namespace kas
