#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>
#include "Halide.h"

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Tuple.hpp"


namespace kas {

inline Halide::Expr ConcretizeSize(const ConcreteConsts& consts, const Size& value) {
    return value.eval<int>(consts.primaryWrapper(), consts.coefficientWrapper());
}

template<typename Storage, auto Mapping>
Halide::Region ConcretizeShape(const ConcreteConsts& consts, const AbstractShape<Storage, Mapping>& shape, bool reverse = true) {
    Halide::Region region;
    for (const auto& s: shape) {
        region.emplace_back(0, ConcretizeSize(consts, s));
    }
    if (reverse) {
        // Because Halide uses column-major, we need to reverse the order of indices.
        std::ranges::reverse(region);
    }
    return region;
}

struct HalideAccess {
    constexpr static int Output = -1;
    int position; // Still, -1 means output tensor.
    std::vector<Halide::Var> outerLoops; // In reverse order.
    Halide::RDom reductionDomain;
    std::vector<Halide::Expr> constraints;
    std::vector<std::vector<Halide::Expr>> inputs; // Inner arrays are in reverse order.
    std::vector<Halide::Expr> output; // In reverse order.

    // Lower to Halide.
    HalideAccess(const ConcreteConsts& consts, const AbstractAccess& access);
};

class HalideGen {
public:
    struct Options {
        enum class AutoScheduler {
            ComputeRoot, Mullapudi2016, Li2018, Adams2019, Anderson2021,
        };
        bool useGPU = true;
        AutoScheduler scheduler = AutoScheduler::Li2018;
    };

private:
    const BindingContext& ctx;
    const TensorView& tensorView;
    Options options;

public:
    static Halide::Target GetHostTarget(bool useGPU);

    static void GuardAutoSchedulers(); // Load the Halide auto scheduler plugins.

    // If lowering a backward pipeline, the output tensor is the gradient of the output tensor.
    Halide::Func lower(std::vector<Halide::ImageParam>& inputTensors, Halide::ImageParam *outputTensor, const HalideAccess& access, std::string_view funcName);

    struct ConcreteShapes {
        std::vector<Halide::Region> inputShapes;
        Halide::Region outputShape;
    };
    ConcreteShapes concretizeShapes(const ConcreteConsts& consts) const;

    using ForwardArgsAndFunc = std::pair<std::vector<Halide::ImageParam>, Halide::Func>;
    // Returns the (input, func) pair.
    ForwardArgsAndFunc createFunc(const ConcreteConsts& consts, const ConcreteShapes& shapes, std::string_view funcName);

    using BackwardArgsAndFuncs = std::pair<std::vector<Halide::ImageParam>, std::vector<Halide::Func>>;
    // Returns the (input (including the gradient of output), gradient funcs) pair.
    BackwardArgsAndFuncs createFuncGrad(const ConcreteConsts& consts, const ConcreteShapes& shapes, std::string_view funcName);

    using ForwardAndBackwardFuncs = std::tuple<std::vector<Halide::ImageParam>, Halide::Func, std::vector<Halide::ImageParam>, std::vector<Halide::Func>>;
    // Returns the forward and backward funcs.
    ForwardAndBackwardFuncs createPipelines(const std::map<std::string, std::size_t>& mappings, std::string_view funcName);

    // Applys auto-schedulers on the funcs.
    using ScheduledPipelins = std::pair<Halide::Pipeline, Halide::Pipeline>;
    static ScheduledPipelins ApplyAutoScheduler(Halide::Func& forwardFunc, std::vector<Halide::Func>& backwardFuncs, const Halide::Target& target, Options::AutoScheduler scheduler, bool verbose);

    static void GenerateFromPipelines(std::vector<Halide::ImageParam>& forwardInputs, std::vector<Halide::ImageParam>& backwardInputs, Halide::Pipeline& forwardPipeline, Halide::Pipeline& backwardPipeline, std::filesystem::path outputPath, std::string_view funcName, const Halide::Target& target);

    inline HalideGen(const BindingContext& ctx, const TensorView& tensorView, Options options):
        ctx { ctx },
        tensorView { tensorView },
        options { std::move(options) }
    {}

    void generate(std::filesystem::path outputPath, std::string_view funcName, const std::map<std::string, std::size_t>& mappings);

    // This is a workaround for reverse indexing caused by Halide's column-major buffers.
    template<typename BufferType>
    struct ReverseAdaptor {
        BufferType content;
        auto operator()(const std::vector<int>& indices) -> decltype(auto) {
            std::vector<int> indicesCopy(indices.rbegin(), indices.rend());
            return content(indicesCopy);
        }
        template<typename... Args>
        requires(Halide::Runtime::AllInts<Args...>::value)
        auto operator()(Args... args) -> decltype(auto) {
            auto helper = [this]<typename... Params>(Params&&... params) -> decltype(auto) {
                return content(std::forward<Params>(params)...);
            };
            return std::apply(helper, ReverseTuple(std::forward_as_tuple(std::forward<Args>(args)...)));
        }
    };

    template<typename T = void, int Dims = Halide::AnyDims>
    using BufferAdaptor = ReverseAdaptor<Halide::Buffer<T, Dims>>;

    template<typename T = void, int Dims = Halide::AnyDims>
    using BufferRefAdaptor = ReverseAdaptor<Halide::Buffer<T, Dims>&>;

};

} // namespace kas
