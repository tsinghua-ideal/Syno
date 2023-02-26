#pragma once

#include <memory>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

class MapReduceOp final: public DimensionImpl {
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
    explicit inline MapReduceOp(std::size_t priority, auto&& domain, MapType mapType, ReduceType reduceType):
        priority { priority },
        domain { std::forward<decltype(domain)>(domain) },
        mapType { mapType },
        reduceType { reduceType }
    {}
    inline const Size& size() const noexcept override { return domain; }
    inline std::size_t initialHash() const noexcept override {
        auto h = priority;
        HashCombine(h, DimensionType::MapReduce);
        return h;
    }
    constexpr DimensionType type() const noexcept override { return DimensionType::MapReduce; }

    inline MapType getMap() const { return mapType; }
    inline ReduceType getReduce() const { return reduceType; }

    inline std::size_t getPriority() const { return priority; }
    inline std::string getName() const {
        return "ri_" + std::to_string(priority);
    }
    std::string whatMap() const;
    std::string whatReduce() const;
    std::string what() const;

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<Interface> GenerateLastLevelMapReduces(const Shape& outputShape, GenerateOptions options);
};

} // namespace kas
