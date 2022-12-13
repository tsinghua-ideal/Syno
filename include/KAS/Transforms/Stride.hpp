#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class StrideShapeOp final: public PrimitiveShapeOp {
public:
    int input;
    int output;
    std::shared_ptr<Size> stride;
    StrideShapeOp(int input, int output, std::shared_ptr<Size> stride);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;

    static std::vector<std::unique_ptr<StrideShapeOp>> generate(const Shape& outputShape);
};

class StrideOp: public RepeatLikePrimitiveOp {
public:
    std::shared_ptr<Size> stride;
    StrideOp(std::shared_ptr<Iterator> parent, std::shared_ptr<Size> stride);
    SingleIteratorValue value(SingleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
