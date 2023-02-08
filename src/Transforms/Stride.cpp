#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <utility>

#include "KAS/Transforms/Stride.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Core/Iterator.hpp"


namespace kas {

StrideShapeOp::StrideShapeOp(std::size_t input, std::size_t output, std::shared_ptr<Size> stride):
    input { input },
    output { output },
    stride { std::move(stride) }
{}

Shape StrideShapeOp::transformShapeInverse(const Shape& outputShape) const {
    KAS_ASSERT(stride->isLegalCoefficient());
    auto out = outputShape[output];
    return outputShape.replace({ output }, { std::make_pair(input, *out * *stride)});
}

void StrideShapeOp::transformTensor(TensorView& tensor) const {
    auto inputIt = tensor[input];
    std::unique_ptr<RepeatLikePrimitiveOp> op { new StrideOp { inputIt, stride } };
    auto outputIt = std::make_shared<Iterator>(IteratorTransform { std::move(op) }, *inputIt->getSize() / *stride);
    tensor.replaceInterface({ input }, { std::make_pair(output, std::move(outputIt)) });
}

std::string StrideShapeOp::description() const {
    std::stringstream ss;
    ss << "Stride " << input << " -> " << output;
    return ss.str();
}

std::vector<std::unique_ptr<StrideShapeOp>> StrideShapeOp::generate(const Shape& outputShape) {
    std::vector<std::unique_ptr<StrideShapeOp>> result;
    for (std::size_t i = 0; i < outputShape.size(); ++i) {
        const Size& size = *outputShape[i];
        auto primary = size.getPrimary();
        if (std::ranges::all_of(primary, [](auto x) { return x == 0; })) {
            // Here, we only allow an axis with primary variable to be strided. TODO: relax this?
            continue;
        }
        auto coefficient = size.getCoefficient();
        for (std::size_t j = 0; j < coefficient.size(); ++j) {
            // Here we take one of the coefficient as stride. If you want more, you can add more StrideShapeOp.
            auto stride = std::make_shared<Size>(primary.size(), coefficient.size());
            stride->getCoefficient()[j] = 1;
            result.emplace_back(std::make_unique<StrideShapeOp>(i, i, stride));
        }
    }
    return result;
}

StrideOp::StrideOp(std::shared_ptr<Iterator> parent, std::shared_ptr<Size> stride):
    RepeatLikePrimitiveOp { std::move(parent) },
    stride { std::move(stride) }
{}

SingleIteratorValue StrideOp::value(SingleIteratorValue output) const {
    auto stride = std::make_shared<ConstValueNode>(this->stride);
    return *stride * *output;
}

} // namespace kas
