#include "KAS/Transforms/MapReduce.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"


namespace kas {

Dimensions MapReduceOp::applyToInterface(const Dimensions& interface) const {
    KAS_UNREACHABLE("No need to apply MapReduceOp to interface.");
}

std::string MapReduceOp::description(const BindingContext& ctx) const {
    return getInput().description(ctx);
}

std::vector<const MapReduceOp *> MapReduceOp::Generate(PrimitiveOpStore& store, const std::vector<const MapReduceOp *>& current, const GenerateOptions& options) {
    const BindingContext& ctx = options.ctx;

    using BaseShapeView = AbstractShape<const std::vector<const MapReduceOp *>&, [](const MapReduceOp *r) -> const Size& { return r->size(); }>;
    const Size& outputSize = options.outputSize;
    Size reductionSize = current.empty() ? outputSize.identity() : BaseShapeView(current).totalSize();

    auto withinFLOPs = [&](const Size& size) {
        std::size_t flops = 0;
        for (const ConcreteConsts& consts: ctx.getAllConsts()) {
            // Conservative approximation.
            flops += outputSize.eval<std::size_t>(consts) * ((reductionSize * size).eval<std::size_t>(consts) - 1);
        }
        return flops <= options.maxFLOPs;
    };

    auto canonical = [&](const Size& size) {
        return current.empty() || Size::LexicographicalLEQ(current.back()->size(), size);
    };

    Allowance allowance = { outputSize * reductionSize, ctx };

    std::vector<const MapReduceOp *> res;
    for (Size size: allowance.enumerateSizes(ctx)) {
        if (canonical(size) && withinFLOPs(size)) {
            // For simplicity, we only use Identity and Mean. TODO: Add more.
            res.push_back(store.get<MapReduceOp>(current.size(), std::move(size), MapType::Identity, ReduceType::Mean));
        }
    }

    return res;
}

} // namespace kas
