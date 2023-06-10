#include <string>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

std::string MapReduceOp::what(MapType type) {
    switch (type) {
        case MapType::Absolute: return "Absolute";
        case MapType::ArcTan:   return "ArcTan";
        case MapType::Exp:      return "Exp";
        case MapType::Log:      return "Log";
        case MapType::Identity: return "Identity";
        case MapType::Inverse:  return "Inverse";
        case MapType::Negative: return "Negative";
        case MapType::ReLU:     return "ReLU";
        case MapType::Sigmoid:  return "Sigmoid";
        case MapType::Sign:     return "Sign";
        case MapType::MapTypeCount: break;
    }
    KAS_UNREACHABLE();
}

std::string MapReduceOp::what(ReduceType type) {
    switch (type) {
        case ReduceType::Sum:     return "Sum";
        case ReduceType::Max:     return "Max";
        case ReduceType::Mean:    return "Mean";
        case ReduceType::Min:     return "Min";
        case ReduceType::Product: return "Product";
        case ReduceType::ReduceTypeCount: break;
    }
    KAS_UNREACHABLE();
}

std::string MapReduceOp::whatMap() const {
    return what(mapType);
}
std::string MapReduceOp::whatReduce() const {
    return what(reduceType);
}

std::string MapReduceOp::what() const {
    return what(mapType) + "+" + what(reduceType);
}

std::string MapReduceOp::description(const BindingContext& ctx) const {
    return Dimension(this).description(ctx);
}

std::vector<const MapReduceOp *> MapReduceOp::Generate(ReductionStore& store, const std::vector<const MapReduceOp *>& current, const GenerateOptions& options) {
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
            res.push_back(store.get(current.size(), std::move(size), MapType::Identity, ReduceType::Mean));
        }
    }

    return res;
}

} // namespace kas
