#pragma once

#include "KAS/Search/PrimitiveShapeOp.hpp"


namespace kas {

class ShareShapeOp: public PrimitiveShapeOp {
public:
    int inputLhs, inputRhs;
    int output;
    ShareShapeOp(int inputLhs, int inputRhs, int output);
    virtual Shape transformShapeInverse(const Shape& input) const override;
    virtual void transformTensor(TensorView& tensor) const override;
};

} // namespace kas
