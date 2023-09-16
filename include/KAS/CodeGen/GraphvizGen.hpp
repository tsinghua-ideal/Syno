#pragma once

#include <filesystem>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/TensorView.hpp"


namespace kas {

namespace detail {
    class GraphvizCode {
    protected:
        std::string code;
    public:
        void generate(const std::filesystem::path& outputPath, std::string_view funcName) const;
        std::string print(std::string_view funcName) const;
    };
};

class GraphvizGen: public detail::GraphvizCode {
public:
    GraphvizGen(const Topmost& inputs, const BindingContext& ctx);
    GraphvizGen(const std::vector<Topmost>& tensors, const BindingContext& ctx);
    GraphvizGen(const TensorView& tensorView, const BindingContext& ctx);
    // Functions to emphasize some Dimension's. TODO
};

class GraphvizDFGGen: public detail::GraphvizCode {
public:
    GraphvizDFGGen(const Subgraphs& subgraphs, const Graph& graph, const BindingContext& ctx);
};

} // namespace kas
