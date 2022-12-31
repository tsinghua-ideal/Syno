#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShareShapeOp final: public PrimitiveShapeOp {
public:
    std::size_t inputLhs, inputRhs;
    std::size_t output;
    ShareShapeOp(std::size_t inputLhs, std::size_t inputRhs, std::size_t output);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;

    static std::vector<std::unique_ptr<ShareShapeOp>> generate(const Shape& outputShape);
};

class ShareOp: public MergeLikePrimitiveOp {
public:
    ShareOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs);
    DoubleIteratorValue value(SingleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
