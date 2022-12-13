#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MergeShapeOp final: public PrimitiveShapeOp {
public:
    int inputMajor, inputMinor;
    int output;
    std::shared_ptr<Size> block;
    MergeShapeOp(int inputMajor, int inputMinor, int output, std::shared_ptr<Size> block);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;

    static std::vector<std::unique_ptr<MergeShapeOp>> generate(const Shape& outputShape);
};

class MergeOp: public MergeLikePrimitiveOp {
public:
    MergeOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs);
    DoubleIteratorValue value(SingleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
