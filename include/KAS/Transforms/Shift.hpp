#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShiftShapeOp final: public PrimitiveShapeOp {
public:
    std::size_t input;
    std::size_t output;
    // Not very sure how to represent shift. TODO
    int shift;
    ShiftShapeOp(std::size_t input, std::size_t output, int shift);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;
    inline std::string type() const override { return "Shift"; }
    std::string description() const override;

    static std::vector<std::unique_ptr<ShiftShapeOp>> generate(const Shape& outputShape);
};

class ShiftOp: public RepeatLikePrimitiveOp {
public:
    int shift;
    ShiftOp(std::shared_ptr<Iterator> parent, int shift);
    IteratorValue value(IteratorValue output) const override;
};

} // namespace kas
