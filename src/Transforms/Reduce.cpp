#include <memory>

#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms/Reduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Core/Iterator.hpp"


namespace kas {

ReduceShapeOp::ReduceShapeOp(int input, std::shared_ptr<Size> size):
    input { input },
    size { std::move(size) }
{}

Shape ReduceShapeOp::transformShapeInverse(const Shape& output) const {
    return output.replace({}, { std::make_pair(input, size) });
}

void ReduceShapeOp::transformTensor(TensorView &tensor) const {
    KAS_ASSERT(tensor.interface.size() > input);
    auto inputIt = tensor[input];
    std::unique_ptr<RepeatLikePrimitiveOp> op { new ReduceOp { inputIt } };
    KAS_ASSERT(size == inputIt->size);
    auto outputIt = std::make_shared<Iterator>(IteratorTransform { std::move(op) }, size);
    tensor.replaceInterface({ input }, {});
    // This is special for Reduce: we need to add it to reducedIterators
    tensor.addManipulation(Manipulation { ReduceManipulation { outputIt } });
}

ReduceOp::ReduceOp(std::shared_ptr<Iterator> parent):
    RepeatLikePrimitiveOp { std::move(parent) }
{}

SingleIteratorValue ReduceOp::value(SingleIteratorValue output) const {
    return std::move(output);
}

} // namespace kas
