#pragma once

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

// This is for convenience. As a python interface, we need easy access to related methods of TensorView.
class Kernel {
protected:
    TensorView tensorView;
    BindingContext& ctx;
    std::shared_ptr<CodeGenContext> cgCtx;
    HalideGen gen;

public:
    template<typename T>
    Kernel(T&& tensorView, BindingContext& ctx, std::shared_ptr<CodeGenContext> cgCtx):
        tensorView { std::forward<T>(tensorView) },
        ctx { ctx },
        cgCtx { std::move(cgCtx) },
        gen { ctx, this->tensorView }
    {}

    std::string toNestedLoops() const;

    void generate(const std::string& path, const std::string& name, HalideGen::Options options, const std::map<std::string, std::size_t>& estimates = {});

    std::vector<std::size_t> getArguments(const std::map<std::string, std::size_t>& mappings) const;

    std::vector<std::vector<std::size_t>> getInputsShapes(const std::map<std::string, std::size_t>& mappings) const;

    std::vector<std::size_t> getOutputShape(const std::map<std::string, std::size_t>& mappings) const;
};

} // namespace kas
