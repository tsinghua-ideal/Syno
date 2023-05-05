#include "KAS/CodeGen/Kernel.hpp"
#include "KAS/CodeGen/GraphvizGen.hpp"


namespace kas {

Kernel::Kernel(const TensorView& tensorView, BindingContext& ctx, const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options):
    tensorView { tensorView },
    ctx { ctx },
    gen { ctx, this->tensorView, std::move(options) }
{
    for (const auto& mappings: allMappings) {
        auto unpadded = ctx.realizeConsts(mappings);
        auto padded = tensorView.computePadding(ctx, unpadded);
        paddedConsts.emplace_back(std::move(unpadded), std::move(padded));
    }
}

std::string Kernel::toNestedLoops() const {
    return tensorView.printNestedLoopsForAll(ctx);
}

void Kernel::generateOperator(const std::string& path, const std::string& name) {
    // Pass padded consts to HalideGen.
    for (std::size_t i = 0; auto&& consts: paddedConsts) {
        gen.generate(path, fmt::format("{}_{}", name, i), consts.padded);
        ++i;
    }
}

void Kernel::generateGraphviz(const std::string& path, const std::string& name) {
    GraphvizGen gen { tensorView, ctx };
    gen.generate(path, name);
}

std::string Kernel::getConsts(std::size_t index) const {
    return paddedConsts.at(index).toString(ctx);
}

std::size_t Kernel::getFLOPs(std::size_t index) const {
    // Because we actually use the padded consts.
    return tensorView.getFLOPs(paddedConsts[index].padded);
}
std::size_t Kernel::getTotalFLOPs() const {
    std::size_t result = 0;
    for (std::size_t i = 0; i < paddedConsts.size(); ++i) {
        result += getFLOPs(i);
    }
    return result;
}

std::vector<std::vector<std::size_t>> Kernel::getInputsShapes(bool padded, std::size_t index) const {
    const auto& consts = padded ? paddedConsts[index].padded : paddedConsts[index].unpadded;
    std::vector<std::vector<std::size_t>> result;
    for (const auto& tensor: tensorView.getUnderlyingTensors()) {
        result.emplace_back(tensor.getShape().eval<std::size_t>(consts));
    }
    return result;
}

std::vector<std::size_t> Kernel::getOutputShape(bool padded, std::size_t index) const {
    const auto& consts = padded ? paddedConsts[index].padded : paddedConsts[index].unpadded;
    return tensorView.getInterfaceShape().eval<std::size_t>(consts);
}

} // namespace kas
