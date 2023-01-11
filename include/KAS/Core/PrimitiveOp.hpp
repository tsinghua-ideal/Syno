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
    // During the search, what we only care is the shape of the tensor. This function transforms the shape of the tensor in a bottom-up way, and ignores the actual semantics of a primitive.
    virtual Shape transformShapeInverse(const Shape& input) const = 0;
    // After the search, when the resulting tensor has a shape that is verified to be eligible, we can build the TensorView, which is a series of transforms on a tensor. The semantics, rather than by this class, are implemented by the PrimitiveOp's defined below. They are inserted by this function to the TensorView.
    virtual void transformTensor(TensorView& tensor) const;
    virtual std::string description() const = 0;
    virtual bool isFinalizeOp() const;
    virtual ~PrimitiveShapeOp() = default;
};

class IdentityShapeOp final: public PrimitiveShapeOp {
public:
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;
    std::string description() const override;
};

// There are 3 kinds of PrimitiveOp's, listed below. Those classes can transform the Iterator, from those that index the output tensor, to forms that index the original tensors. So this is also kind of bottom-up.

using SingleIteratorValue = std::shared_ptr<IteratorValue>;
using DoubleIteratorValue = std::pair<SingleIteratorValue, SingleIteratorValue>;

// By repeat-like, we refer to the primitives that have one input iterator and one output iterator.
class RepeatLikePrimitiveOp {
public:
    std::shared_ptr<Iterator> parent;
    // std::weak_ptr<Iterator> child; // Actually not needed
    RepeatLikePrimitiveOp(std::shared_ptr<Iterator> parent);
    // Compute input iterator from output iterator
    virtual SingleIteratorValue value(SingleIteratorValue output) const = 0;
    virtual ~RepeatLikePrimitiveOp() = default;
};

// By split-like, we refer to the primitives that have one input iterator and two output iterators.
class SplitLikePrimitiveOp {
public:
    std::shared_ptr<Iterator> parent;
    std::weak_ptr<Iterator> childLhs, childRhs; // Actually only here children are needed
    // The std::weak_ptr<Iterator> is impossible to be given during construction. You can just give a default constructed pointer, for example, std::weak_ptr<Iterator>(). The parameters are just a reminder: do not forget to set them after Iterator is constructed.
    SplitLikePrimitiveOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs);
    // Compute input iterator from output iterators
    virtual SingleIteratorValue value(DoubleIteratorValue output) const = 0;
    virtual ~SplitLikePrimitiveOp() = default;
};

// By merge-like, we refer to the primitives that have two input iterators and one output iterator.
class MergeLikePrimitiveOp {
public:
    std::shared_ptr<Iterator> parentLhs, parentRhs;
    // std::weak_ptr<Iterator> child; // Actually not needed
    MergeLikePrimitiveOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs);
    // Compute output iterators from input iterator
    virtual DoubleIteratorValue value(SingleIteratorValue output) const = 0;
    virtual ~MergeLikePrimitiveOp() = default;
};

} // namespace kas
