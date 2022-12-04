#include "KAS/Search/ShapeNode.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/Iterator.hpp"
#include <memory>


namespace kas {

ShapeNode::ShapeNode(std::shared_ptr<ShapeNode> child, std::unique_ptr<PrimitiveShapeOp> shapeOp):
    shape { shapeOp->transformShapeInverse(child->shape) },
    child { std::move(child) },
    shapeOp { std::move(shapeOp) }
{}

TensorView ShapeNode::buildTensor() const {
    auto tensor = std::make_shared<PureTensor>(shape);
    TensorView tensorView { tensor->shared_from_this() };
    auto curr = this;
    while (curr && curr->child) {
        curr->shapeOp->transformTensor(tensorView);
        curr = curr->child.get();
    }
    return tensorView;
}

} // namespace kas
