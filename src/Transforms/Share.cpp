#include "KAS/Transforms/Share.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Search/PrimitiveShapeOp.hpp"
#include "KAS/Utils/Common.hpp"
#include <algorithm>
#include <utility>


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
    auto op = MergeLikePrimitiveOp { tensor[inputLhs], tensor[inputRhs] };
    auto it = std::make_shared<Iterator>(IteratorTransform { op });
    tensor.replaceInterface({ inputLhs, inputRhs }, {
        std::make_pair(output, it)
    });
}

} // namespace kas
