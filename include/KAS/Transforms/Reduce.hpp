#pragma once

#include <memory>

#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ReduceShapeOp final: public PrimitiveShapeOp {
public:
    int input;
    std::shared_ptr<Size> size;
    ReduceManipulation::Type type;
    ReduceShapeOp(int input, std::shared_ptr<Size> size, ReduceManipulation::Type type);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;

    static std::vector<std::unique_ptr<ReduceShapeOp>> generate(const Shape& outputShape);
};

class ReduceOp: public RepeatLikePrimitiveOp {
public:
    ReduceOp(std::shared_ptr<Iterator> parent);
    SingleIteratorValue value(SingleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
