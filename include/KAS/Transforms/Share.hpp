#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShareShapeOp: public PrimitiveShapeOp {
public:
    int inputLhs, inputRhs;
    int output;
    ShareShapeOp(int inputLhs, int inputRhs, int output);
    virtual Shape transformShapeInverse(const Shape& input) const override;
    virtual void transformTensor(TensorView& tensor) const override;
};

class ShareOp: public MergeLikePrimitiveOp {
public:
    ShareOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs);
    virtual DoubleIteratorValue value(SingleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
