#pragma once

#include <variant>

#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/PrimitiveShapeOp.hpp"


namespace kas {

using IteratorTransform = std::variant<RepeatLikePrimitiveOp, SplitLikePrimitiveOp, MergeLikePrimitiveOp, TensorStub>;

class Iterator {
public:
    IteratorTransform parent;
    Iterator(IteratorTransform parent);
};

} // namespace kas
