#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

inline std::size_t ShareOp::Input::hash() const noexcept {
    std::size_t h = static_cast<std::size_t>(type());
    HashCombine(h, op->output.hash());
    HashCombine(h, order);
    return h;
}

std::pair<IteratorValue, IteratorValue> ShareOp::value(const IteratorValue& output) const {
    return { output, output };
}

std::vector<std::unique_ptr<ShareOp>> ShareOp::Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options) {
    Allowance allowance { Size::Product(ShapeView(outputShape)), options.ctx };
    std::vector<std::unique_ptr<ShareOp>> result;
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
