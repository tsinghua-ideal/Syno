#pragma once

#include <memory>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

// Contains multiple finalization options.
class FinalizeOp {
    using Tensors = std::vector<Interface>;
    std::vector<Tensors> options;

public:
    std::unique_ptr<TensorView> buildTensorView() const;

    struct GenerateOptions {
        const BindingContext& ctx;
        const Shape& desired;
    };
    static std::vector<FinalizeOp> Generate(const Interface& outputShape, GenerateOptions options);
};

} // namespace kas
