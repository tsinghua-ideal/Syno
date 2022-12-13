#include <memory>
#include <set>

#include "KAS/Search/ShapeNode.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

ShapeNode::ShapeNode(std::shared_ptr<ShapeNode> child, std::unique_ptr<PrimitiveShapeOp> shapeOp):
    shape { shapeOp->transformShapeInverse(child->shape) },
    child { std::move(child) },
    shapeOp { std::move(shapeOp) }
{}

TensorView ShapeNode::buildTensorView(std::size_t tensorId) const {
    auto tensor = std::make_shared<PureTensor>(tensorId, shape);
    TensorView tensorView { tensor->shared_from_this() };
    auto curr = this;
    while (curr && curr->child) {
        curr->shapeOp->transformTensor(tensorView);
        curr = curr->child.get();
    }
    tensorView.finishConstruction();
    return tensorView;
}

} // namespace kas
