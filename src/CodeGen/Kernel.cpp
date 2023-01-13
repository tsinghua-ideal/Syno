#include "KAS/CodeGen/Kernel.hpp"


namespace kas {

std::string Kernel::toNestedLoops() const {
    return tensorView.printNestedLoops(ctx);
}
std::string Kernel::description() const {
    return repr.description();
}
void Kernel::generate(const std::string& path, const std::string& name, HalideGen::Options options) {
    gen.generate(path, name, options);
}

} // namespace kas
