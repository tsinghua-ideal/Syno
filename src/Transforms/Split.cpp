#include <memory>
#include <sstream>
#include <utility>

#include "KAS/Transforms/Split.hpp"
#include "KAS/Core/Iterator.hpp"


namespace kas {

SplitShapeOp::SplitShapeOp(std::size_t input, std::size_t outputMajor, std::size_t outputMinor):
    input { input },
    outputMajor { outputMajor },
    outputMinor { outputMinor }
{}

Shape SplitShapeOp::transformShapeInverse(const Shape& outputShape) const {
    auto outputMajorSize = outputShape[outputMajor];
    auto outputMinorSize = outputShape[outputMinor];
    block = outputMinorSize;
    return outputShape.replace({ outputMajor, outputMinor }, {
        std::make_pair(input, *outputMajorSize * *outputMinorSize)
    });
}

void SplitShapeOp::transformTensor(TensorView& tensor) const {
    auto inputIt = tensor[input];
    std::shared_ptr<SplitLikePrimitiveOp> op { new SplitOp { inputIt, std::weak_ptr<Iterator>(), std::weak_ptr<Iterator>() } };
    auto outputMajorIt = std::make_shared<Iterator>(IteratorTransform { op }, *inputIt->getSize() / *block);
    auto outputMinorIt = std::make_shared<Iterator>(IteratorTransform { op }, block);
    op->childLhs = outputMajorIt;
    op->childRhs = outputMinorIt;
    tensor.replaceInterface({ input }, {
        std::make_pair(outputMajor, std::move(outputMajorIt)),
        std::make_pair(outputMinor, std::move(outputMinorIt))
    });
}

SplitOp::SplitOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs):
    SplitLikePrimitiveOp { parent, childLhs, childRhs }
{}

SingleIteratorValue SplitOp::value(DoubleIteratorValue output, const BindingContext& ctx) const {
    auto [outputMajor, outputMinor] = std::move(output);
    auto block = std::make_shared<ConstValueNode>(childRhs.lock()->getSize());
    return *(*outputMajor * *block) + *outputMinor;
}

} // namespace kas
