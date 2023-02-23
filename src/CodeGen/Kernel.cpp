#include "KAS/CodeGen/Kernel.hpp"


namespace kas {

std::string Kernel::toNestedLoops() const {
    return tensorView.printNestedLoops(ctx);
}

void Kernel::generate(const std::string& path, const std::string& name, HalideGen::Options options, const std::map<std::string, std::size_t>& mappings) {
    gen.generate(path, name, mappings, options);
}

std::vector<std::vector<std::size_t>> Kernel::getInputsShapes(const std::map<std::string, std::size_t>& mappings) const {
    auto consts = ctx.realizeConsts(mappings);
    std::vector<std::vector<std::size_t>> result;
    for (const auto& tensor: tensorView.getUnderlyingTensors()) {
        result.emplace_back(tensor.getShape().eval<std::size_t>(consts));
    }
    return result;
}

std::vector<std::size_t> Kernel::getOutputShape(const std::map<std::string, std::size_t>& mappings) const {
    auto consts = ctx.realizeConsts(mappings);
    return tensorView.getShape().eval<std::size_t>(consts);
}

} // namespace kas
