#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

std::pair<IteratorValue, IteratorValue> ShareOp::value(const IteratorValue& output) const {
    return { output, output };
}

std::vector<const ShareOp *> ShareOp::Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options) {
    Allowance allowance { Size::Product(ShapeView(outputShape)), options.ctx };
    std::vector<const ShareOp *> result;
    if (outputShape.size() < options.dimUpperBound) {
        for (auto&& dim: outputShape) {
            if (allowance.withinAllowance(dim.size())) {
                result.emplace_back(store.get<ShareOp>(dim));
            }
        }
    }
    return result;
}

} // namespace kas
