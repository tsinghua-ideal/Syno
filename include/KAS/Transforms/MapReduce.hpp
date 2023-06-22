#pragma once

#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MapReduceOp final: public PrimitiveOp {
public:
    static constexpr DimensionType Type = DimensionType::MapReduce;
    using MapType = MapReduce::MapType;
    using ReduceType = MapReduce::ReduceType;

private:
    MapReduce reduction;

    MapType getMap() const { return reduction.getMap(); }
    ReduceType getReduce() const { return reduction.getReduce(); }
    std::size_t getPriority() const { return reduction.getPriority(); }
    const Size& size() const { return reduction.size(); }

public:
    MapReduceOp(std::size_t priority, auto&& domain, MapType mapType, ReduceType reduceType):
        PrimitiveOp { Color::None },
        reduction { priority, std::forward<decltype(domain)>(domain), mapType, reduceType }
    {}
    MapReduceOp(const MapReduce&) = delete;
    MapReduceOp(MapReduce&&) = delete;

    constexpr DimensionType getType() const noexcept override { return Type; }
    std::size_t initialHash() const noexcept override { return reduction.hash(); }
    std::size_t opHash() const noexcept override { return initialHash(); }

    const MapReduce *getRaw() const { return &reduction; }
    Dimension getInput() const { return &reduction; }

    Dimensions applyToInterface(const Dimensions& interface) const override;

    bool operator==(const MapReduceOp& other) const noexcept {
        return getMap() == other.getMap() && getReduce() == other.getReduce() && getPriority() == other.getPriority() && size() == other.size();
    }

    std::string description(const BindingContext& ctx) const override;

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
        Size outputSize;
        std::size_t maxFLOPs;
        std::size_t maximumReductions;
    };
    static std::vector<const MapReduceOp *> Generate(PrimitiveOpStore& store, const std::vector<const MapReduceOp *>& current, const GenerateOptions& options);
};

} // namespace kas
