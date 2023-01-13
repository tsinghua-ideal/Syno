#pragma once

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/Representation.hpp"
#include "KAS/Core/Tensor.hpp"
#include <utility>


namespace kas {

// This is for convenience. As a python interface, we need easy access to related methods of TensorView.
class Kernel {
protected:
    TensorView tensorView;
    BindingContext& ctx;
    std::shared_ptr<CodeGenContext> cgCtx;
    HalideGen gen;
    Representation repr;

public:
    template<typename Tt, typename Tr>
    Kernel(Tt&& tensorView, BindingContext& ctx, std::shared_ptr<CodeGenContext> cgCtx, Tr&& repr):
        tensorView { std::forward<Tt>(tensorView) },
        ctx { ctx },
        cgCtx { std::move(cgCtx) },
        gen { ctx, this->tensorView },
        repr { std::forward<Tr>(repr) }
    {}

    std::string toNestedLoops() const;

    std::string description() const;

    void generate(const std::string& path, const std::string& name, HalideGen::Options options);
};

} // namespace kas
