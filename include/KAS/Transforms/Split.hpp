#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class SplitShapeOp: public PrimitiveShapeOp {
public:
    int input;
    int outputMajor, outputMinor;
    // A bit ugly, but we have to maintain the size here.
    mutable std::shared_ptr<Size> block;
    SplitShapeOp(int input, int outputMajor, int outputMinor);
    virtual Shape transformShapeInverse(const Shape& input) const override;
    virtual void transformTensor(TensorView& tensor) const override;
};

class SplitOp: public SplitLikePrimitiveOp {
public:
    SplitOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs);
    virtual SingleIteratorValue value(DoubleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
