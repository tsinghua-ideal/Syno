#pragma once

#include <memory>

#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ReduceShapeOp: public PrimitiveShapeOp {
public:
    int input;
    std::shared_ptr<Size> size;
    ReduceManipulation::Type type;
    ReduceShapeOp(int input, std::shared_ptr<Size> size, ReduceManipulation::Type type);
    virtual Shape transformShapeInverse(const Shape& input) const override;
    virtual void transformTensor(TensorView& tensor) const override;
};

class ReduceOp: public RepeatLikePrimitiveOp {
public:
    ReduceOp(std::shared_ptr<Iterator> parent);
    virtual SingleIteratorValue value(SingleIteratorValue output) const override;
};

} // namespace kas
