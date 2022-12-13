#pragma once

#include <memory>
#include <optional>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class UnfoldShapeOp final: public PrimitiveShapeOp {
public:
    int input;
    int outputOriginal, outputWindow;
    // A bit ugly, but we have to maintain the size here.
    mutable std::shared_ptr<Size> windowSize;
    UnfoldShapeOp(int input, int outputOriginal, int outputWindow);
    Shape transformShapeInverse(const Shape& outputShape) const override;
    void transformTensor(TensorView& tensor) const override;

    static std::vector<std::unique_ptr<UnfoldShapeOp>> generate(const Shape& outputShape);
};

class UnfoldOp: public SplitLikePrimitiveOp {
public:
    std::shared_ptr<Size> window;
    UnfoldOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs);
    SingleIteratorValue value(DoubleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
