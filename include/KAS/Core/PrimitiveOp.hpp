#pragma once

#include <utility>
#include <memory>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

class Iterator;
class IteratorValue;

class PrimitiveShapeOp {
public:
    virtual Shape transformShapeInverse(const Shape& input) const = 0;
    virtual void transformTensor(TensorView& tensor) const = 0;
};

using SingleIteratorValue = std::shared_ptr<IteratorValue>;
using DoubleIteratorValue = std::pair<SingleIteratorValue, SingleIteratorValue>;

class RepeatLikePrimitiveOp {
public:
    std::shared_ptr<Iterator> parent;
    // std::weak_ptr<Iterator> child; // Actually not needed
    RepeatLikePrimitiveOp(std::shared_ptr<Iterator> parent);
    // Compute input iterator from output iterator
    virtual SingleIteratorValue value(SingleIteratorValue output) const = 0;
};

class SplitLikePrimitiveOp {
public:
    std::shared_ptr<Iterator> parent;
    std::weak_ptr<Iterator> childLhs, childRhs; // Actually only here children are needed
    // The std::weak_ptr<Iterator> is impossible to be given during construction. You can just give a default constructed pointer, for example, std::weak_ptr<Iterator>(). The parameters are just a reminder: do not forget to set them after Iterator is constructed.
    SplitLikePrimitiveOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs);
    // Compute input iterator from output iterators
    virtual SingleIteratorValue value(DoubleIteratorValue output) const = 0;
};

class MergeLikePrimitiveOp {
public:
    std::shared_ptr<Iterator> parentLhs, parentRhs;
    // std::weak_ptr<Iterator> child; // Actually not needed
    MergeLikePrimitiveOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs);
    // Compute output iterators from input iterator
    virtual DoubleIteratorValue value(SingleIteratorValue output) const = 0;
};

} // namespace kas
