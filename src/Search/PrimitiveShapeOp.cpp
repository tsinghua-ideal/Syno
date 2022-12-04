#include "KAS/Search/PrimitiveShapeOp.hpp"


namespace kas {

RepeatLikePrimitiveOp::RepeatLikePrimitiveOp(std::shared_ptr<Iterator> parent):
    parent { std::move(parent) }
{}

SplitLikePrimitiveOp::SplitLikePrimitiveOp(std::shared_ptr<Iterator> parent):
    parent { std::move(parent) }
{}

MergeLikePrimitiveOp::MergeLikePrimitiveOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs):
    parentLhs { std::move(parentLhs) },
    parentRhs { std::move(parentRhs) }
{}

} // namespace kas
