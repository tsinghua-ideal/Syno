#pragma once

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

class MapReduce: public DimensionImpl {
public:
    enum class MapType {
        Absolute,
        ArcTan,
        Exp,
        Log,
        Identity,
        Inverse,
        Negative,
        ReLU,
        Sigmoid,
        Sign,
        MapTypeCount,
    };
    static std::string what(MapType);
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
    // This decides the order in which MapReduce is applied.
    std::size_t priority;
    Size domain;

    MapType mapType;
    ReduceType reduceType;

public:
    MapReduce(std::size_t priority, const Size& domain, MapType mapType, ReduceType reduceType);
    const Size& size() const noexcept final override { return domain; }
    std::size_t hash() const noexcept final override;
    constexpr DimensionType type() const noexcept final override { return DimensionType::MapReduce; }
    void accept(DimVisitor& visitor) const final override;
    const Color& getColor() const final override { return Color::None; }

    MapType getMap() const { return mapType; }
    ReduceType getReduce() const { return reduceType; }

    std::size_t getPriority() const { return priority; }

    std::string whatMap() const;
    std::string whatReduce() const;
};

} // namespace kas
