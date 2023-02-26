#pragma once

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

class DimensionStore;

// There are 3 kinds of `PrimitiveOp`'s, listed below. Those classes can transform `Dimension`s, from those that index the output tensor, to forms that index the original tensors. So this is also kind of bottom-up.

// By repeat-like, we refer to the primitives that have one input iterator and one output iterator.
class RepeatLikePrimitiveOp: public DimensionImpl {
public:
    Dimension output;
    RepeatLikePrimitiveOp(auto&& output):
        output { std::forward<decltype(output)>(output) }
    {}
    virtual IteratorValue value(const IteratorValue& value) const = 0;
};
struct NextRepeatLike {
    Dimension input;
};

// By split-like, we refer to the primitives that have one input iterator and two output iterators.
class SplitLikePrimitiveOp: public DimensionImpl {
public:
    Dimension outputLhs, outputRhs;
    SplitLikePrimitiveOp(auto&& outputLhs, auto&& outputRhs):
        outputLhs { std::forward<decltype(outputLhs)>(outputLhs) },
        outputRhs { std::forward<decltype(outputRhs)>(outputRhs) }
    {}
    virtual IteratorValue value(const IteratorValue& leftValue, const IteratorValue& rightValue) const = 0;
};
struct NextSplitLike {
    Dimension input;
};

enum class Order: bool {
    Left = false,
    Right = true,
};
// By merge-like, we refer to the primitives that have two input iterators and one output iterator.
class MergeLikePrimitiveOp: public DimensionImpl {
public:
    Dimension output;
    Order order;
    MergeLikePrimitiveOp(auto&& output, Order order):
        output { std::forward<decltype(output)>(output) },
        order { order }
    {}
    virtual IteratorValue value(const IteratorValue& value) const = 0;
};
struct NextMergeLike {
    Dimension inputLhs, inputRhs;
};

} // namespace kas
