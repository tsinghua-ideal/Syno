#pragma once

#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class SplitShapeOp final: public PrimitiveShapeOp {
public:
    std::size_t input;
    std::size_t outputMajor, outputMinor;
    // A bit ugly, but we have to maintain the size here.
    mutable std::shared_ptr<Size> block;
    SplitShapeOp(std::size_t input, std::size_t outputMajor, std::size_t outputMinor);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;

    struct GenerateOptions {
        int dimLowerBound;
    };
    static std::vector<std::unique_ptr<SplitShapeOp>> generate(const Shape& outputShape, GenerateOptions options);
};

class SplitOp: public SplitLikePrimitiveOp {
public:
    SplitOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs);
    SingleIteratorValue value(DoubleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
