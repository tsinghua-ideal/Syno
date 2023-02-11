#pragma once

#include <memory>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class ShareShapeOp final: public PrimitiveShapeOp {
public:
    std::size_t inputLhs, inputRhs;
    std::size_t output;
    ShareShapeOp(std::size_t inputLhs, std::size_t inputRhs, std::size_t output);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;
    inline std::string type() const override { return "Share"; }
    std::string description() const override;

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimUpperBound;
    };
    static std::vector<std::unique_ptr<ShareShapeOp>> generate(const Shape& outputShape, GenerateOptions options);
};

class ShareOp: public MergeLikePrimitiveOp {
public:
    ShareOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs);
    DoubleIteratorValue value(IteratorValue output) const override;
};

} // namespace kas
