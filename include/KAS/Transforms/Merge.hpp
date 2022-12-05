#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MergeShapeOp: public PrimitiveShapeOp {
public:
    int inputMajor, inputMinor;
    int output;
    std::shared_ptr<Size> block;
    MergeShapeOp(int inputMajor, int inputMinor, int output, std::shared_ptr<Size> block);
    virtual Shape transformShapeInverse(const Shape& input) const override;
    virtual void transformTensor(TensorView& tensor) const override;
};

class MergeOp: public MergeLikePrimitiveOp {
public:
    MergeOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs);
    virtual DoubleIteratorValue value(SingleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
