#include "KAS/Transforms/Reduce.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"


namespace kas {

bool ReduceOp::canApplyToInterface(const GraphHandle& interface) const {
    KAS_CRITICAL("You cannot decide whether a ReduceOp can be applied to an interface without the StageStore.");
}

GraphHandle ReduceOp::applyToInterface(const GraphHandle& interface) const {
    return interface.insert1(getInput());
}

std::string ReduceOp::description(const BindingContext& ctx) const {
    return getInput().description(ctx);
}
std::string ReduceOp::descendantsDescription(const BindingContext& ctx) const {
    return description(ctx);
}

std::vector<const ReduceOp *> ReduceOp::Generate(PrimitiveOpStore& store, const std::vector<const ReduceOp *>& current, const GenerateOptions& options) {
    const BindingContext& ctx = options.ctx;

    using BaseShapeView = AbstractShape<const std::vector<const ReduceOp *>&, [](const ReduceOp *r) -> const Size& { return r->size(); }>;
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

    std::vector<const ReduceOp *> res;
    for (Size size: allowance.enumerateSizes(ctx)) {
        if (canonical(size) && withinFLOPs(size)) {
            // For simplicity, we only use Identity and Mean. TODO: Add more.
            res.push_back(store.get<ReduceOp>(current.size(), std::move(size), MapType::Identity, ReduceType::Mean));
        }
    }

    return res;
}

} // namespace kas
