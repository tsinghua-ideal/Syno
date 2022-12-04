#pragma once

#include <utility>
#include <variant>
#include <memory>
#include <tuple>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

class Iterator;

class PrimitiveShapeOp {
public:
    virtual Shape transformShapeInverse(const Shape& input) const = 0;
    virtual void transformTensor(TensorView& tensor) const = 0;
};

class RepeatLikePrimitiveOp {
public:
    std::shared_ptr<Iterator> parent;
    RepeatLikePrimitiveOp(std::shared_ptr<Iterator> parent);
};

class SplitLikePrimitiveOp {
public:
    std::shared_ptr<Iterator> parent;
    SplitLikePrimitiveOp(std::shared_ptr<Iterator> parent);
};

class MergeLikePrimitiveOp {
public:
    std::shared_ptr<Iterator> parentLhs, parentRhs;
    MergeLikePrimitiveOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs);
};

} // namespace kas
