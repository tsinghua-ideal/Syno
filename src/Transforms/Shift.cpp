#include <memory>
#include <sstream>
#include <utility>

#include "KAS/Transforms/Shift.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Core/Iterator.hpp"


namespace kas {

ShiftShapeOp::ShiftShapeOp(std::size_t input, std::size_t output, int shift):
    input { input },
    output { output },
    shift { shift }
{}

Shape ShiftShapeOp::transformShapeInverse(const Shape& outputShape) const {
    return outputShape;
}

void ShiftShapeOp::transformTensor(TensorView& tensor) const {
    auto inputIt = tensor[input];
    std::unique_ptr<RepeatLikePrimitiveOp> op { new ShiftOp { inputIt, shift } };
    auto outputIt = std::make_shared<Iterator>(IteratorTransform { std::move(op) }, inputIt->getSize());
    tensor.replaceInterface({ input }, { std::make_pair(output, std::move(outputIt)) });
}

ShiftOp::ShiftOp(std::shared_ptr<Iterator> parent, int shift):
    RepeatLikePrimitiveOp { std::move(parent) },
    shift { shift }
{}

SingleIteratorValue ShiftOp::value(SingleIteratorValue output, const BindingContext& ctx) const {
    auto imm = std::make_shared<ImmediateValueNode>(shift);
    auto size = std::make_shared<ConstValueNode>(parent->getSize());
    return *(*(*output + *imm) + *size) % *size;
}

} // namespace kas
