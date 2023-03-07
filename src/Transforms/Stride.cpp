#include <algorithm>

#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Stride.hpp"


namespace kas {

std::size_t StrideOp::initialHash() const noexcept {
    std::size_t h = static_cast<std::size_t>(Type);
    HashCombine(h, stride);
    return h;
}

IteratorValue StrideOp::value(const IteratorValue& output) const {
    auto stride = ConstValueNode::Create(this->stride);
    return stride * output;
}

std::vector<StrideOp *> StrideOp::Generate(DimensionStore& store, const Interface& outputShape) {
    std::vector<StrideOp *> result;
    for (std::size_t i = 0; i < outputShape.size(); ++i) {
        const Size& size = outputShape[i].size();
        auto primary = size.getPrimary();
        if (std::ranges::all_of(primary, [](auto x) { return x == 0; })) {
            // Here, we only allow an axis with primary variable to be strided. TODO: relax this?
            continue;
        }
        auto coefficient = size.getCoefficient();
        for (std::size_t j = 0; j < coefficient.size(); ++j) {
            // Here we take one of the coefficient as stride. If you want more, you can add more StrideShapeOp.
            auto stride = Size(primary.size(), coefficient.size());
            stride.getCoefficient()[j] = 1;
            result.emplace_back(store.get<StrideOp>(outputShape[i], stride));
        }
    }
    return result;
}

} // namespace kas
