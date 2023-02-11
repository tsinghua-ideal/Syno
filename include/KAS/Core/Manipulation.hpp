#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string_view>
#include <variant>


namespace kas {

class Iterator;

// Represents a map-reduce op.
class Manipulation {
    friend class HalideGen;

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
    std::shared_ptr<Iterator> iterator;
    MapType mapType;
    ReduceType reduceType;

public:
    // Illegal. Must be set before use.
    std::size_t iteratorVariableId = std::numeric_limits<std::size_t>::max();

    Manipulation(std::shared_ptr<Iterator> iterator, MapType mapType, ReduceType reduceType);

    std::shared_ptr<Iterator> getIterator() const;

    std::string whatMap() const;
    std::string whatReduce() const;
    std::string what() const;
};

} // namespace kas
