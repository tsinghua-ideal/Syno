#include <memory>
#include <sstream>

#include "KAS/Transforms/Merge.hpp"
#include "KAS/Core/Iterator.hpp"


namespace kas {

MergeShapeOp::MergeShapeOp(int inputMajor, int inputMinor, int output, std::shared_ptr<Size> block):
    inputMajor { inputMajor },
    inputMinor { inputMinor },
    output { output },
    block { std::move(block) }
{}

Shape MergeShapeOp::transformShapeInverse(const Shape& outputShape) const {
    return outputShape.replace({ output }, {
        std::make_pair(inputMajor, *outputShape[output] / *block),
        std::make_pair(inputMinor, block)
    });
}

void MergeShapeOp::transformTensor(TensorView& tensor) const {
    auto inputMajorIt = tensor[inputMajor];
    auto inputMinorIt = tensor[inputMinor];
    std::unique_ptr<MergeLikePrimitiveOp> op { new MergeOp { inputMajorIt, inputMinorIt } };
    auto outputIt = std::make_shared<Iterator>(IteratorTransform { std::move(op) }, *inputMajorIt->getSize() * *inputMinorIt->getSize());
    tensor.replaceInterface({ inputMajor, inputMinor }, { std::make_pair(output, std::move(outputIt)) });
}

MergeOp::MergeOp(std::shared_ptr<Iterator> parentMajor, std::shared_ptr<Iterator> parentMinor):
    MergeLikePrimitiveOp { std::move(parentMajor), std::move(parentMinor) }
{}

DoubleIteratorValue MergeOp::value(SingleIteratorValue output, const BindingContext& ctx) const {
    auto block = parentRhs->getSize()->toString(ctx);
    std::stringstream ss;
    ss << "(" << output->content << ")/(" << block << ")";
    auto major = std::make_shared<IteratorValue>(ss.str());
    ss.str("");
    ss << "(" << output->content << ")%(" << block << ")";
    auto minor = std::make_shared<IteratorValue>(ss.str());
    return std::make_pair(std::move(major), std::move(minor));
}

} // namespace kas
