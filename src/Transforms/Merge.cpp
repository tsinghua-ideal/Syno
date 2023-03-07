#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Merge.hpp"


namespace kas {

const Size& MergeOp::Input::size() const noexcept {
    switch (order) {
    case Order::Left:
        return getDerivedOp<MergeOp>()->majorSize;
    case Order::Right:
        return getDerivedOp<MergeOp>()->minorSize;
    }
}

std::size_t MergeOp::initialHash() const noexcept {
    std::size_t h = static_cast<std::size_t>(Type);
    HashCombine(h, minorSize);
    return h;
}

std::pair<IteratorValue, IteratorValue> MergeOp::value(const IteratorValue& output) const {
    auto block = ConstValueNode::Create(this->minorSize);
    return { output / block, output % block };
}

std::vector<MergeOp *> MergeOp::Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options) {
    const auto& ctx = options.ctx;
    auto primaryCount = ctx.getPrimaryCount(), coefficientCount = ctx.getCoefficientCount();
    std::vector<MergeOp *> res;
    if (outputShape.size() < options.dimUpperBound) {
        for (std::size_t i = 0; i < outputShape.size(); ++i) {
            const auto& size = outputShape[i].size();
            if (size.getPrimaryPowersSum() == 0) {
                // Here we only split dimensions that have >= 1 primary variables. TODO: split dimensions with only coefficients.
                continue;
            }
            auto primary = size.getPrimary();
            auto coefficient = size.getCoefficient();
            for (std::size_t primaryIndex = 0; primaryIndex < primaryCount; ++primaryIndex) {
                int primaryDim = primary[primaryIndex];
                if (primaryDim >= 1) {
                    auto primaryRes = Size(primaryCount, coefficientCount);
                    // Splitting out power of one is enough. For more, use more MergeOp's.
                    primaryRes.getPrimary()[primaryIndex] = 1;
                    auto canBeDivided = size.canBeDividedBy(primaryRes);
                    if (canBeDivided.has_value() && canBeDivided.value() != Size::Trait::One) {
                        res.emplace_back(store.get<MergeOp>(outputShape[i], primaryRes));
                    }
                    for (std::size_t coefficientIndex = 0; coefficientIndex < coefficientCount; ++coefficientIndex) {
                        std::size_t coefficientDim = coefficient[coefficientIndex];
                        if (coefficientDim != 0) {
                            auto coefRes = Size(primaryRes);
                            // Here we simply split out the coefficient in half. TODO: better sampling.
                            if (canBeDivided.has_value()) {
                                coefRes.getCoefficient()[coefficientIndex] = coefficientDim > 0 ? ((coefficientDim + 1) / 2) : ((coefficientDim - 1) / 2);
                            } else {
                                coefRes.getCoefficient()[coefficientIndex] = coefficientDim > 0 ? ((coefficientDim + 1) / 2) : coefficientDim;
                            }
                            canBeDivided = size.canBeDividedBy(coefRes);
                            if (canBeDivided.has_value() && canBeDivided.value() != Size::Trait::One) {
                                res.emplace_back(store.get<MergeOp>(outputShape[i], coefRes));
                            }
                        }
                    }
                }
            }
        }
    }
    return res;
}

} // namespace kas
