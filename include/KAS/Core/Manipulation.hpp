#pragma once

#include <memory>
#include <variant>


namespace kas {

class Iterator;

class ReduceManipulation {
public:
    enum class Type {
        Sum,
        Max,
        Mean,
        Min,
        Product,
    };
    static std::string what(Type);
    Type type;
    std::shared_ptr<Iterator> iterator;
    ReduceManipulation(std::shared_ptr<Iterator> iterator, Type type);
    std::string what() const;
};

class MapManipulation {
public:
    enum class Type {
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
    static std::string what(Type);
    Type type;
    MapManipulation(Type type);
    std::string what() const;
};

using Manipulation = std::variant<ReduceManipulation, MapManipulation>;

} // namespace kas
