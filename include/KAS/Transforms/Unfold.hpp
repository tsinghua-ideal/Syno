#pragma once

#include <memory>
#include <optional>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class UnfoldShapeOp final: public PrimitiveShapeOp {
public:
    std::size_t input;
    std::size_t outputOriginal, outputWindow;
    // A bit ugly, but we have to maintain the size here.
    mutable std::shared_ptr<Size> windowSize;
    UnfoldShapeOp(std::size_t input, std::size_t outputOriginal, std::size_t outputWindow);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;
    inline std::string type() const override { return "Unfold"; }
    std::string description() const override;

    struct GenerateOptions {
        const BindingContext& ctx;
        std::size_t dimLowerBound;
    };
    static std::vector<std::unique_ptr<UnfoldShapeOp>> generate(const Shape& outputShape, GenerateOptions options);
};

class UnfoldOp: public SplitLikePrimitiveOp {
public:
    std::shared_ptr<Size> originalSize;
    UnfoldOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs, std::shared_ptr<Size> originalSize);
    IteratorValue value(DoubleIteratorValue output) const override;
};

} // namespace kas
