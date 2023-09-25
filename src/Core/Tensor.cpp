#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Ranges.hpp"

namespace kas {

const std::vector<Tensor>& Tensor::inputs() const {
    return inner->inputs;
}
const std::vector<Dimension>& Tensor::output() const {
    return inner->output;
}
const std::vector<const Reduce *>& Tensor::reductions() const {
    return inner->reductions;
}

void Tensor::adjustLayout(const std::vector<Dimension> *expectedOutput, const std::vector<const Reduce *> *expectedReductions) {
    if (expectedOutput) {
        KAS_ASSERT(DimensionSetEqual(output(), *expectedOutput));
        getOutput() = *expectedOutput;
    }
    if (expectedReductions) {
        // TODO: check equal.
        getReductions() = *expectedReductions;
    }
}

Tensor Tensor::clone(const std::map<Tensor, Tensor>& oldToNew) const {
    return TensorImpl::CreateView(
        ranges::to<std::vector<Tensor>>(
            inputs()
            | std::views::transform([&](const Tensor& t) {
                return oldToNew.at(t);
            })
        ),
        output(),
        reductions()
    );
}

Size Tensor::getNumElements(const BindingContext& ctx) const {
    // TODO: this has some redundant code with IR.cpp.
    auto numelOuter = std::transform_reduce(
        output().begin(), output().end(), // outer loops.
        Size::Identity(ctx), std::multiplies<>(),
        [](const Dimension& dim) -> const Size& { return dim.size(); }
    );
    auto numel = std::transform_reduce(
        reductions().begin(), reductions().end(), // inner loops.
        numelOuter, std::multiplies<>(),
        [&](const Reduce *reduce) -> const Size& { return reduce->size(); }
    );
    return numel;
}

std::size_t Tensor::getFLOPs(const BindingContext& ctx, const ConcreteConsts& consts) const {
    auto numel = getNumElements(ctx);
    auto instsPerAddition = hasContraction() ? std::max<std::size_t>(inputs().size() - 1, 1) : 1;
    return numel.eval<std::size_t>(consts) * instsPerAddition;
}

std::size_t Tensor::getFLOPs(const BindingContext& ctx) const {
    auto numel = getNumElements(ctx);
    auto instsPerAddition = hasContraction() ? std::max<std::size_t>(inputs().size() - 1, 1) : 1;
    std::size_t flops = 0;
    for (const ConcreteConsts& consts: ctx.getAllConsts()) {
        flops += numel.eval<std::size_t>(consts) * instsPerAddition;
    }
    return flops;
}

ConstrainedGraph Tensor::buildConstrainedGraph(const Graph& graph) const {
    return ConstrainedGraph::Builder(graph)
        .addTop(inputs() | std::views::transform(&Tensor::output) | std::views::join)
        .addBottom(output())
        .addBottom(reductions())
        .build();
}

std::string Tensor::toString(const BindingContext& ctx) const {
    auto outputString = ShapeView(output()).toString(ctx);
    if (isInputTensor()) {
        return outputString;
    }
    return fmt::format("({} -> {})", fmt::join(inputs() | std::views::transform([&](const Tensor& t) { return t.toString(ctx); }), ", "), outputString);
}

std::string Tensor::debugToString() const {
    return BindingContext::ApplyDebugPublicCtx(&Tensor::toString, *this);
}

} // namespace kas
