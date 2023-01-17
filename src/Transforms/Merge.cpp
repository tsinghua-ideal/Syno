#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>

#include "KAS/Transforms/Merge.hpp"
#include "KAS/Core/Iterator.hpp"


namespace kas {

MergeShapeOp::MergeShapeOp(std::size_t inputMajor, std::size_t inputMinor, std::size_t output, std::shared_ptr<Size> block):
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

std::string MergeShapeOp::description() const {
    std::stringstream ss;
    ss << "Merge " << inputMajor << ", " << inputMinor << " -> " << output;
    return ss.str();
}

std::vector<std::unique_ptr<MergeShapeOp>> MergeShapeOp::generate(const Shape& outputShape, GenerateOptions options) {
    const auto& ctx = options.ctx;
    auto primaryCount = ctx.getPrimaryCount(), coefficientCount = ctx.getCoefficientCount();
    std::vector<std::unique_ptr<MergeShapeOp>> res;
    if (outputShape.size() < options.dimUpperBound) {
        for (std::size_t i = 0; i < outputShape.size(); ++i) {
            const auto& size = *outputShape[i];
            if (size.getPrimaryPowersSum() == 0) {
                // Here we only split dimensions that have >= 1 primary variables. TODO: split dimensions with only coefficients.
                continue;
            }
            auto primary = size.getPrimary();
            auto coefficient = size.getCoefficient();
            for (std::size_t primaryIndex = 0; primaryIndex < primaryCount; ++primaryIndex) {
                int primaryDim = primary[primaryIndex];
                if (primaryDim >= 1) {
                    auto primaryRes = std::make_shared<Size>(primaryCount, coefficientCount);
                    // Splitting out power of one is enough. For more, use more MergeOp's.
                    primaryRes->getPrimary()[primaryIndex] = 1;
                    auto canBeDivided = size.canBeDividedBy(*primaryRes);
                    if (canBeDivided.has_value() && canBeDivided.value() != Size::Trait::One) {
                        res.emplace_back(std::make_unique<MergeShapeOp>(i, i + 1, i, primaryRes));
                    }
                    for (int coefficientIndex = 0; coefficientIndex < coefficientCount; ++coefficientIndex) {
                        std::size_t coefficientDim = coefficient[coefficientIndex];
                        if (coefficientDim != 0) {
                            auto coefRes = std::make_shared<Size>(*primaryRes);
                            // Here we simply split out the coefficient in half. TODO: better sampling.
                            if (canBeDivided.has_value()) {
                                coefRes->getCoefficient()[coefficientIndex] = coefficientDim > 0 ? ((coefficientDim + 1) / 2) : ((coefficientDim - 1) / 2);
                            } else {
                                coefRes->getCoefficient()[coefficientIndex] = coefficientDim > 0 ? ((coefficientDim + 1) / 2) : coefficientDim;
                            }
                            canBeDivided = size.canBeDividedBy(*coefRes);
                            if (canBeDivided.has_value() && canBeDivided.value() != Size::Trait::One) {
                                res.emplace_back(std::make_unique<MergeShapeOp>(i, i + 1, i, std::move(coefRes)));
                            }
                        }
                    }
                }
            }
        }
    }
    return res;
}

MergeOp::MergeOp(std::shared_ptr<Iterator> parentMajor, std::shared_ptr<Iterator> parentMinor):
    MergeLikePrimitiveOp { std::move(parentMajor), std::move(parentMinor) }
{}

DoubleIteratorValue MergeOp::value(SingleIteratorValue output) const {
    auto block = std::make_shared<ConstValueNode>(parentRhs->getSize());
    return std::make_pair(*output / *block, *output % *block);
}

} // namespace kas
