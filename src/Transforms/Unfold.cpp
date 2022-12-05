#include <memory>
#include <sstream>
#include <utility>

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Transforms/Unfold.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Core/Iterator.hpp"


namespace kas {

UnfoldShapeOp::UnfoldShapeOp(int input, int outputOriginal, int outputWindow):
    input { input },
    outputOriginal { outputOriginal },
    outputWindow { outputWindow }
{}

Shape UnfoldShapeOp::transformShapeInverse(const Shape& outputShape) const {
    auto windowSize = outputShape[outputWindow];
    KAS_ASSERT(windowSize->isCoefficient());
    this->windowSize = windowSize;
    return outputShape.replace({ outputOriginal, outputWindow }, { std::make_pair(input, outputShape[outputOriginal]) });
}

void UnfoldShapeOp::transformTensor(TensorView& tensor) const {
    KAS_ASSERT(windowSize); // transformShapeInverse() must be called before this!
    auto inputIt = tensor[input];
    std::shared_ptr<SplitLikePrimitiveOp> op { new UnfoldOp { inputIt, std::weak_ptr<Iterator>(), std::weak_ptr<Iterator>() } };
    auto outputMajor = std::make_shared<Iterator>(IteratorTransform { op }, inputIt->getSize());
    auto outputMinor = std::make_shared<Iterator>(IteratorTransform { op }, windowSize);
    op->childLhs = outputMajor;
    op->childRhs = outputMinor;
    tensor.replaceInterface({ input }, { std::make_pair(outputOriginal, std::move(outputMajor)), std::make_pair(outputWindow, std::move(outputMinor)) });
}

UnfoldOp::UnfoldOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs):
    SplitLikePrimitiveOp { parent, childLhs, childRhs }
{}

SingleIteratorValue UnfoldOp::value(DoubleIteratorValue output, const BindingContext& ctx) const {
    auto [outputMajor, outputMinor] = std::move(output);
    std::stringstream ss;
    ss << "(" << outputMajor->content << "+" << outputMinor->content << "-" << childRhs.lock()->getSize()->toString(ctx) << "/2)";
    return std::make_shared<IteratorValue>(ss.str());
}

} // namespace kas
