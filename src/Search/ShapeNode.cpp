#include <memory>
#include <set>

#include "KAS/Search/ShapeNode.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

ShapeNode::ShapeNode(Shape&& shape, bool isFinal):
    shape { std::move(shape) },
    isFinal { isFinal }
{}

ShapeNode::Next::Next(std::unique_ptr<PrimitiveShapeOp> shapeOp):
    shapeOp { std::move(shapeOp) }
{}

} // namespace kas
