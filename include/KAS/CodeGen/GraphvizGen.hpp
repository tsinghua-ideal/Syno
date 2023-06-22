#pragma once

#include <filesystem>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

class GraphvizGen {
    std::string code;
public:
    GraphvizGen(const std::vector<Dimension>& inputs, const BindingContext& ctx);
    GraphvizGen(const std::vector<std::vector<Dimension>>& tensors, const BindingContext& ctx);
    GraphvizGen(const TensorView& tensorView, const BindingContext& ctx);
    void generate(std::filesystem::path outputDirectory, std::string_view funcName) const;
    std::string print(std::string_view funcName) const;
    // Functions to emphasize some Dimensions. TODO
};

} // namespace kas
