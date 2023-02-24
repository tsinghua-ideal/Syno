#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Merge.hpp"


namespace kas {

std::size_t MergeOp::initialHash() const noexcept {
    std::size_t seed = std::hash<std::string>{}("Merge");
    boost::hash_combine(seed, block);
    return seed;
}

IteratorValue MergeOp::value(const IteratorValue& output) const {
    auto block = ConstValueNode::Create(this->block);
    switch (order) {
    case Order::Left:
        return output / block;
    case Order::Right:
        return output % block;
    }
}

std::vector<std::pair<Dimension, Dimension>> MergeOp::Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options) {
    const auto& ctx = options.ctx;
    auto primaryCount = ctx.getPrimaryCount(), coefficientCount = ctx.getCoefficientCount();
    std::vector<std::pair<Dimension, Dimension>> res;
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
                        res.emplace_back(store.get<MergeOp>(outputShape[i], Order::Left, primaryRes), store.get<MergeOp>(outputShape[i], Order::Right, primaryRes));
                    }
                    for (std::size_t coefficientIndex = 0; coefficientIndex < coefficientCount; ++coefficientIndex) {
                        std::size_t coefficientDim = coefficient[coefficientIndex];
                        if (coefficientDim != 0) {
                            auto coefRes = std::make_shared<Size>(primaryRes);
                            // Here we simply split out the coefficient in half. TODO: better sampling.
                            if (canBeDivided.has_value()) {
                                coefRes->getCoefficient()[coefficientIndex] = coefficientDim > 0 ? ((coefficientDim + 1) / 2) : ((coefficientDim - 1) / 2);
                            } else {
                                coefRes->getCoefficient()[coefficientIndex] = coefficientDim > 0 ? ((coefficientDim + 1) / 2) : coefficientDim;
                            }
                            canBeDivided = size.canBeDividedBy(*coefRes);
                            if (canBeDivided.has_value() && canBeDivided.value() != Size::Trait::One) {
                                res.emplace_back(store.get<MergeOp>(outputShape[i], Order::Left, coefRes), store.get<MergeOp>(outputShape[i], Order::Right, coefRes));
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
