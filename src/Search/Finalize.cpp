#include <memory>

#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Finalize.hpp"


namespace kas {

std::unique_ptr<TensorView> FinalizeOp::buildTensorView() const {
    return std::make_unique<TensorView>(tensors);
}

std::vector<FinalizeOp> FinalizeOp::Generate(const ColoredInterface& outputShape, GenerateOptions options) {
    // TODO
    return { FinalizeOp { outputShape.toInterface() } };
}

} // namespace kas
