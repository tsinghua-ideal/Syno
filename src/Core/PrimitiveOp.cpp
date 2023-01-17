#include <memory>

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

bool PrimitiveShapeOp::isFinalizeOp() const {
    return false;
}

Shape IdentityShapeOp::transformShapeInverse(const Shape &outputShape) const {
    return outputShape; 
}

void IdentityShapeOp::transformTensor(TensorView& tensor) const {
    KAS_CRITICAL("The root node should not call this function.");
}

std::string IdentityShapeOp::description() const {
    KAS_CRITICAL("The root node should not call this function.");
}

RepeatLikePrimitiveOp::RepeatLikePrimitiveOp(std::shared_ptr<Iterator> parent):
    parent { std::move(parent) }
{}

SplitLikePrimitiveOp::SplitLikePrimitiveOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs):
    parent { std::move(parent) },
    childLhs { std::move(childLhs) },
    childRhs { std::move(childRhs) }
{}

MergeLikePrimitiveOp::MergeLikePrimitiveOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs):
    parentLhs { std::move(parentLhs) },
    parentRhs { std::move(parentRhs) }
{}

} // namespace kas
