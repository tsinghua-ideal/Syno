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

    auto shape = ReductionShapeView(current);
    Size reductionSize = current.empty() ? Size::Identity(ctx) : shape.totalSize();

    auto withinFLOPs = [&](const Size& size) {
        for (const ConcreteConsts& consts: ctx.getAllConsts()) {
            // This actually has nothing to do with FLOPs.
            auto rate = (reductionSize * size / options.maxRDomSizeBase).evalFraction<std::size_t>(consts) / options.maxRDomSizeMultiplier;
            if (rate > 1) {
                return false;
            }
        }
        return true;
    };

    std::vector<const ReduceOp *> res;
    for (Size size: options.allowance.enumerateSizes()) {
        if (withinFLOPs(size)) {
            // For simplicity, we only use Sum. TODO: Add more.
            res.push_back(store.get<ReduceOp>(std::move(size), ReduceType::Sum));
        }
    }

    return res;
}

} // namespace kas
