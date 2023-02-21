#pragma once

#include <boost/container_hash/hash_fwd.hpp>
#include <memory>

#include "KAS/Core/Dimension.hpp"


namespace kas {

class Node;
class StageStore;

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
        boost::hash_combine(h, "MapReduce");
        return h;
    }
    constexpr DimensionType type() const noexcept override { return DimensionType::MapReduce; }

    std::string whatMap() const;
    std::string whatReduce() const;
    std::string what() const;

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<std::unique_ptr<Node>> Generate(StageStore& store, const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
