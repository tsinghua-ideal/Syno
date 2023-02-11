#include <algorithm>
#include <memory>
#include <utility>

#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

ShareShapeOp::ShareShapeOp(std::size_t inputLhs, std::size_t inputRhs, std::size_t output):
    inputLhs { inputLhs },
    inputRhs { inputRhs },
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
    KAS_ASSERT(tensor.getInterfaceIterators().size() > output);
    KAS_ASSERT(inputLhs != inputRhs);
    auto lhs = tensor[inputLhs], rhs = tensor[inputRhs];
    std::unique_ptr<MergeLikePrimitiveOp> op { new ShareOp { lhs, rhs } };
    KAS_ASSERT(*lhs->getSize() == *rhs->getSize());
    std::shared_ptr<Size> size = lhs->getSize();
    auto it = std::make_shared<Iterator>(IteratorTransform { std::move(op) }, std::move(size));
    tensor.replaceInterface({ inputLhs, inputRhs }, {
        std::make_pair(output, std::move(it))
    });
}

std::string ShareShapeOp::description() const {
    std::stringstream ss;
    ss << "Share " << inputLhs << ", " << inputRhs << " -> " << output;
    return ss.str();
}

std::vector<std::unique_ptr<ShareShapeOp>> ShareShapeOp::generate(const Shape& outputShape, GenerateOptions options) {
    Allowance allowance { outputShape.totalSize(), options.ctx };
    std::vector<std::unique_ptr<ShareShapeOp>> result;
    if (outputShape.size() < options.dimUpperBound) {
        for (std::size_t i = 0; i < outputShape.size(); ++i) {
            if (allowance.withinAllowance(*outputShape[i])) {
                // New dimension is put at 0, as the outer loop.
                result.emplace_back(std::make_unique<ShareShapeOp>(0, i + 1, i));
            }
        }
    }
    return result;
}

ShareOp::ShareOp(std::shared_ptr<Iterator> parentLhs, std::shared_ptr<Iterator> parentRhs):
    MergeLikePrimitiveOp { std::move(parentLhs), std::move(parentRhs) }
{}

DoubleIteratorValue ShareOp::value(IteratorValue output) const {
    return { output, output };
}

} // namespace kas
