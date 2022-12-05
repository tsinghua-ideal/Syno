#include <algorithm>
#include <memory>
#include <utility>

#include "KAS/Transforms/Share.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

ShareShapeOp::ShareShapeOp(int inputLhs, int inputRhs, int output):
    inputLhs { std::min(inputLhs, inputRhs) },
    inputRhs { std::max(inputLhs, inputRhs) },
    output { output }
{}

Shape ShareShapeOp::transformShapeInverse(const Shape& outputShape) const {
    KAS_ASSERT(outputShape.size() > output);
    KAS_ASSERT(inputLhs != inputRhs);
    return outputShape.replace({ output }, {
        std::make_pair(inputLhs, outputShape[output]),
        std::make_pair(inputRhs, outputShape[output])
    });
}

void ShareShapeOp::transformTensor(TensorView &tensor) const {
    KAS_ASSERT(tensor.interface.size() > output);
    KAS_ASSERT(inputLhs != inputRhs);
    auto lhs = tensor[inputLhs], rhs = tensor[inputRhs];
    std::unique_ptr<MergeLikePrimitiveOp> op { new ShareOp { lhs, rhs } };
    KAS_ASSERT(lhs->size == rhs->size);
    std::shared_ptr<Size> size = lhs->size;
    auto it = std::make_shared<Iterator>(IteratorTransform { std::move(op) }, std::move(size));
    tensor.replaceInterface({ inputLhs, inputRhs }, {
        std::make_pair(output, it)
    });
}

ShareOp::ShareOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs):
    MergeLikePrimitiveOp { std::move(parentLhs), std::move(parentRhs) }
{}

DoubleIteratorValue ShareOp::value(SingleIteratorValue output) const {
    return { output, output };
}

} // namespace kas
