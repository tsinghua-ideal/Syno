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

// remap: index of dimension in tensor -> index of iterator in interface, not necessarily complete
TensorView ShapeNode::buildTensorView(const std::vector<int>& remap) const {
    std::vector<std::shared_ptr<Size>> realShape;
    realShape.reserve(shape.size());
    std::vector<int> proxy(shape.size(), -1);
    int proxied = 0;
    std::set<int> remapped;
    for (int i = 0; i < remap.size(); ++i) {
        const int remapTarget = remap[i];
        realShape.push_back(shape[remapTarget]);
        proxy[remapTarget] = i;
        ++proxied;
        remapped.insert(remapTarget);
    }
    int current = remap.size();
    for (int i = 0; i < shape.size(); ++i) {
        if (remapped.find(i) == remapped.end()) {
            realShape.push_back(shape[i]);
            proxy[i] = current;
            ++proxied;
            ++current;
        }
    }
    KAS_ASSERT(proxied == shape.size());
    auto tensor = std::make_shared<PureTensor>(Shape { realShape });
    TensorView tensorView { tensor->shared_from_this(), std::move(proxy) };
    auto curr = this;
    while (curr && curr->child) {
        curr->shapeOp->transformTensor(tensorView);
        curr = curr->child.get();
    }
    return tensorView;
}

} // namespace kas
