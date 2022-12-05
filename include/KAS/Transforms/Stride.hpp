#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class StrideShapeOp: public PrimitiveShapeOp {
public:
    int input;
    int output;
    std::shared_ptr<Size> stride;
    StrideShapeOp(int input, int output, std::shared_ptr<Size> stride);
    virtual Shape transformShapeInverse(const Shape& input) const override;
    virtual void transformTensor(TensorView& tensor) const override;
};

class StrideOp: public RepeatLikePrimitiveOp {
public:
    std::shared_ptr<Size> stride;
    StrideOp(std::shared_ptr<Iterator> parent, std::shared_ptr<Size> stride);
    virtual SingleIteratorValue value(SingleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
