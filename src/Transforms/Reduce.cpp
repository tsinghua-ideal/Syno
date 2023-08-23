#include "KAS/Transforms/Reduce.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"


namespace kas {

std::size_t ReduceOp::getMultiplicity(const std::vector<Dimension>& interface) const {
    std::size_t multiplicity = 0;
    for (const Dimension& dim: interface) {
        switch (dim.type()) {
        case DimensionType::Iterator: break; // just simple output.
        case DimensionType::Reduce: {
            const Reduce& reduction = dim.as<Reduce>();
            if (reduction.size() == domain) {
                ++multiplicity;
            }
            break;
        }
        default:
            // There are other Op's.
            KAS_CRITICAL("Cannot get multiplicity of ReduceOp in transformed interface.");
        }
    }
    return multiplicity;
}

std::size_t ReduceOp::getMultiplicity(const GraphHandle& interface) const {
    // Here we assume that we are doing ReductionStage first.
    // So no other Op's.
    KAS_ASSERT(interface.getExpansions().empty());
    return getMultiplicity(interface.getDimensions());
}

bool ReduceOp::canApplyToInterface(const GraphHandle& interface) const {
    KAS_CRITICAL("You cannot decide whether a ReduceOp can be applied to an interface without the StageStore.");
}

GraphHandle ReduceOp::applyToInterface(const GraphHandle& interface) const {
    return interface.insert1(getInput(getMultiplicity(interface)));
}

std::string ReduceOp::description(const BindingContext& ctx) const {
    // Do not append hash. Because size is enough to identify.
    // [<size>]@<type>
    return fmt::format("[{}]@Reduce", domain.toString(ctx));
}
std::string ReduceOp::descendantsDescription(const BindingContext& ctx) const {
    return description(ctx);
}

std::vector<const ReduceOp *> ReduceOp::Generate(PrimitiveOpStore& store, const std::vector<const Reduce *>& current, const GenerateOptions& options) {
    const BindingContext& ctx = options.ctx;

    using BaseShapeView = AbstractShape<const std::vector<const Reduce *>&, [](const Reduce *r) -> const Size& { return r->size(); }>;
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

    Allowance allowance = { outputSize * reductionSize, ctx };

    std::vector<const ReduceOp *> res;
    for (Size size: allowance.enumerateSizes(ctx)) {
        if (withinFLOPs(size)) {
            // For simplicity, we only use Mean. TODO: Add more.
            res.push_back(store.get<ReduceOp>(std::move(size), ReduceType::Mean));
        }
    }

    return res;
}

} // namespace kas
