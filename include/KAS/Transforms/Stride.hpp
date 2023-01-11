#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class StrideShapeOp final: public PrimitiveShapeOp {
public:
    std::size_t input;
    std::size_t output;
    std::shared_ptr<Size> stride;
    StrideShapeOp(std::size_t input, std::size_t output, std::shared_ptr<Size> stride);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;
    std::string description() const override;

    static std::vector<std::unique_ptr<StrideShapeOp>> generate(const Shape& outputShape);
};

class StrideOp: public RepeatLikePrimitiveOp {
public:
    std::shared_ptr<Size> stride;
    StrideOp(std::shared_ptr<Iterator> parent, std::shared_ptr<Size> stride);
    SingleIteratorValue value(SingleIteratorValue output) const override;
};

} // namespace kas
