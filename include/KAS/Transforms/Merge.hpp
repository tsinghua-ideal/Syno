#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class MergeShapeOp final: public PrimitiveShapeOp {
public:
    std::size_t inputMajor, inputMinor;
    std::size_t output;
    std::shared_ptr<Size> block;
    MergeShapeOp(std::size_t inputMajor, std::size_t inputMinor, std::size_t output, std::shared_ptr<Size> block);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;
    std::string description() const override;

    static std::vector<std::unique_ptr<MergeShapeOp>> generate(const Shape& outputShape);
};

class MergeOp: public MergeLikePrimitiveOp {
public:
    MergeOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs);
    DoubleIteratorValue value(SingleIteratorValue output) const override;
};

} // namespace kas
