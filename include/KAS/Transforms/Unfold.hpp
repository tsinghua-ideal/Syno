#pragma once

#include <memory>
#include <optional>

#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class UnfoldShapeOp: public PrimitiveShapeOp {
public:
    int input;
    int outputOriginal, outputWindow;
    // A bit ugly, but we have to maintain the size here.
    mutable std::shared_ptr<Size> windowSize;
    UnfoldShapeOp(int input, int outputOriginal, int outputWindow);
    virtual Shape transformShapeInverse(const Shape& input) const override;
    virtual void transformTensor(TensorView& tensor) const override;
};

class UnfoldOp: public SplitLikePrimitiveOp {
public:
    std::shared_ptr<Size> window;
    UnfoldOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs, std::shared_ptr<Size> window);
    virtual SingleIteratorValue value(DoubleIteratorValue output, const BindingContext& ctx) const override;
};

} // namespace kas
