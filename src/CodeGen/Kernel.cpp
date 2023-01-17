#include "KAS/CodeGen/Kernel.hpp"


namespace kas {

std::string Kernel::toNestedLoops() const {
    return tensorView.printNestedLoops(ctx);
}

void Kernel::generate(const std::string& path, const std::string& name, HalideGen::Options options, const std::map<std::string, std::size_t>& estimates) {
    ctx.applyEstimates(estimates);
    gen.generate(path, name, options);
}

std::vector<std::size_t> Kernel::getArguments(const std::map<std::string, std::size_t>& mappings) const {
    return ctx.getKernelArguments(mappings);
}

std::vector<std::vector<std::size_t>> Kernel::getInputsShapes(const std::map<std::string, std::size_t>& mappings) const {
    std::vector<std::vector<std::size_t>> result;
    for (const auto& tensor: tensorView.getUnderlyingTensors()) {
        result.emplace_back(ctx.evaluateShape(tensor->getShapeRef(), mappings));
    }
    return result;
}

} // namespace kas
