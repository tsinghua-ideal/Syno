#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <utility>

#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Transforms/Unfold.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Core/Iterator.hpp"


namespace kas {

UnfoldShapeOp::UnfoldShapeOp(std::size_t input, std::size_t outputOriginal, std::size_t outputWindow):
    input { input },
    outputOriginal { outputOriginal },
    outputWindow { outputWindow }
{}

Shape UnfoldShapeOp::transformShapeInverse(const Shape& outputShape) const {
    auto windowSize = outputShape[outputWindow];
    KAS_ASSERT(windowSize->isLegalCoefficient());
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

std::string UnfoldShapeOp::description() const {
    std::stringstream ss;
    ss << "Unfold " << input << " -> " << outputOriginal << ", " << outputWindow;
    return ss.str();
}

std::vector<std::unique_ptr<UnfoldShapeOp>> UnfoldShapeOp::generate(const Shape& outputShape, GenerateOptions options) {
    std::vector<std::unique_ptr<UnfoldShapeOp>> result;
    if (outputShape.size() > options.dimLowerBound) {
        std::vector<std::size_t> generals;
        std::vector<std::size_t> windows;
        auto meta = options.ctx.getCoefficientMetadata();
        auto coefficientCount = options.ctx.getCoefficientCount();
        for (std::size_t i = 0; i < outputShape.size(); ++i) {
            const auto& size = *outputShape[i];
            auto p = size.getPrimary();
            if (std::any_of(p.begin(), p.end(), [](auto i) { return i != 0; })) {
                // Here, we only allow an axis with primary variable to be unfolded. TODO: relax this?
                generals.emplace_back(i);
            } else {
                auto coefficient = size.getCoefficient();
                bool isEven = false;
                for (std::size_t i = 0; i < coefficientCount; ++i) {
                    if (coefficient[i] != 0 && !meta[i].isOdd) {
                        isEven = true;
                    }
                }
                if (!isEven) {
                    windows.emplace_back(i);
                }
            }
        }
        for (auto general: generals) {
            for (auto window: windows) {
                KAS_ASSERT(general != window);
                result.emplace_back(std::make_unique<UnfoldShapeOp>(std::min(general, window), general, window));
            }
        }
    }
    return result;
}

UnfoldOp::UnfoldOp(std::shared_ptr<Iterator> parent, std::weak_ptr<Iterator> childLhs, std::weak_ptr<Iterator> childRhs):
    SplitLikePrimitiveOp { parent, childLhs, childRhs }
{}

SingleIteratorValue UnfoldOp::value(DoubleIteratorValue output) const {
    auto [outputMajor, outputMinor] = std::move(output);
    auto kernel = std::make_shared<ConstValueNode>(childRhs.lock()->getSize());
    auto one = std::make_shared<ImmediateValueNode>(1);
    auto two = std::make_shared<ImmediateValueNode>(2);
    return *(*outputMajor + *outputMinor) - *(*(*kernel - *one) / *two);
}

} // namespace kas
