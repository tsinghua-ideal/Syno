#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShiftShapeOp: public PrimitiveShapeOp {
public:
    int input;
    int output;
    // Not very sure how to represent shift. TODO
    int shift;
    ShiftShapeOp(int input, int output, int shift);
    virtual Shape transformShapeInverse(const Shape& input) const override;
    virtual void transformTensor(TensorView& tensor) const override;
};

class ShiftOp: public RepeatLikePrimitiveOp {
public:
    int shift;
    ShiftOp(std::shared_ptr<Iterator> parent, int shift);
    virtual SingleIteratorValue value(SingleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
