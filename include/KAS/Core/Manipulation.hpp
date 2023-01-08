#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <variant>


namespace kas {

class Iterator;

// Represents a map-reduce op.
class Manipulation {
public:
    enum class MapType {
        Absolute,
        ArcTan,
        Exp,
        Log,
        Identity,
        Inverse,
        Mask,
        Negative,
        ReLU,
        Sigmoid,
        Sign,
    };
    static std::string_view what(MapType);
    enum class ReduceType {
        Sum,
        Max,
        Mean,
        Min,
        Product,
    };
    static std::string_view what(ReduceType);

protected:
    MapType mapType;
    ReduceType reduceType;
    std::shared_ptr<Iterator> iterator;

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
