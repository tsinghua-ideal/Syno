#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

// Contains multiple finalization options.
class FinalizeOp {
    std::vector<Interface> tensors;

public:
    FinalizeOp(auto&& tensors): tensors { std::forward<decltype(tensors)>(tensors) } {}
    std::unique_ptr<TensorView> buildTensorView() const;

    static std::size_t CountSuccesses;
    static std::size_t CountFailures;
    static std::size_t CountLegalFinalizations;
    static std::size_t CountConflictingColors;
    struct GenerateOptions {
        const BindingContext& ctx;
        const Shape& desired;
        std::size_t maximumTensors;
    };
    static std::vector<FinalizeOp> Generate(const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options);
};

} // namespace kas
