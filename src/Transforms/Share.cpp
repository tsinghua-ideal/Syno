#include <vector>

#include <fmt/core.h>

#include "KAS/Core/Dimension.hpp"


namespace kas {

DoubleIteratorValue ShareOp::value(const IteratorValue& output) const {
    return { output, output };
}

std::vector<std::pair<Dimension, Dimension>> ShareOp::Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options) {
    std::vector<std::pair<Dimension, Dimension>> result;
    if (outputShape.size() < options.dimUpperBound) {
        for (auto&& dim: outputShape) {
            result.emplace_back(store.get<ShareOp>(dim, FirstOrSecond::First), store.get<ShareOp>(dim, FirstOrSecond::Second));
        }
    }
    // Allowance allowance { outputShape.totalSize(), options.ctx };
    // std::vector<std::unique_ptr<ShareShapeOp>> result;
    // if (outputShape.size() < options.dimUpperBound) {
    //     for (std::size_t i = 0; i < outputShape.size(); ++i) {
    //         if (allowance.withinAllowance(*outputShape[i])) {
    //             // New dimension is put at 0, as the outer loop.
    //             result.emplace_back(std::make_unique<ShareShapeOp>(0, i + 1, i));
    //         }
    //     }
    // }
    return result;
}

} // namespace kas
