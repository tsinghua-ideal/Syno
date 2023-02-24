#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

IteratorValue ShareOp::value(const IteratorValue& output) const {
    return output;
}

std::vector<std::pair<Dimension, Dimension>> ShareOp::Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options) {
    Allowance allowance { Size::Product(ShapeView(outputShape)), options.ctx };
    std::vector<std::pair<Dimension, Dimension>> result;
    if (outputShape.size() < options.dimUpperBound) {
        for (auto&& dim: outputShape) {
            if (allowance.withinAllowance(dim.size())) {
                result.emplace_back(store.get<ShareOp>(dim, Order::Left), store.get<ShareOp>(dim, Order::Right));
            }
        }
    }
    return result;
}

} // namespace kas
