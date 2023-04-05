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
        static inline std::vector<int> RegionToVector(const Halide::Region& region) {
            std::vector<int> result;
            for (const auto& r: region) {
                result.push_back(*Halide::Internal::as_const_int(r.extent));
            }
            return result;
        }
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

    // For test use only.
    struct Realization {
        Halide::Pipeline pipeline;
        HalideGen::BufferAdaptor<float> trial;
        Halide::Pipeline backwardPipeline;
        std::vector<HalideGen::BufferAdaptor<float>> backwardTrials;
    };
    template<bool DoInitialization = true, typename... InputInitializers>
    Realization performTrial(const std::map<std::string, std::size_t>& mappings, auto&& funcName, bool createStaticLibrary, auto&& outputGradInitializer, InputInitializers&&... inputInitializers) {
        auto consts = ctx.realizeConsts(mappings);
        auto shapes = concretizeShapes(consts);
        auto [inputs, func, backwardInputs, backwardFuncs] = createPipelines(mappings, std::forward<decltype(funcName)>(funcName));

        // Initialize input buffers.
        KAS_ASSERT(tensorView.getUnderlyingTensors().size() == sizeof...(inputInitializers));
        std::tuple<decltype(inputInitializers)...> initializerTuple = { std::forward<decltype(inputInitializers)>(inputInitializers)... };
        std::vector<Halide::Buffer<float>> inputBuffers;
        std::vector<Halide::Buffer<float>> inputGradsBuffers;
        auto setter = [&]<std::size_t i>() {
            std::vector<int> inputBufferShape = ConcreteShapes::RegionToVector(shapes.inputShapes.at(i));
            inputGradsBuffers.emplace_back(inputBufferShape);
            inputBuffers.emplace_back(inputBufferShape);
            auto& inputBuffer = inputBuffers.back();
            if constexpr (DoInitialization) {
                auto proxy = HalideGen::BufferRefAdaptor<float>(inputBuffer);
                inputBuffer.for_each_element(ReverseArguments(std::bind_front(std::get<i>(initializerTuple), std::ref(proxy))));
            }
            inputs.at(i).set(inputBuffer);
            backwardInputs.at(i).set(inputBuffer);
        };
        [&]<std::size_t... i>(std::index_sequence<i...>) {
            (setter.template operator()<i>(), ...);
        }(std::make_index_sequence<sizeof...(InputInitializers)>());

        // Compute the forward result.
        auto target = HalideGen::GetHostTarget(options.useGPU);
        auto [pipeline, backwardPipeline] = HalideGen::ApplyAutoScheduler(func, backwardFuncs, target, options.scheduler, true);

        if (createStaticLibrary) {
            HalideGen::GenerateFromPipelines(inputs, backwardInputs, pipeline, backwardPipeline, "./kernel_" + std::string(funcName), funcName, target);
        }

        auto outputBufferShape = ConcreteShapes::RegionToVector(shapes.outputShape);
        Halide::Buffer<float> forwardTrialResult;
        try {
            forwardTrialResult = pipeline.realize(outputBufferShape, target);
        } catch (const Halide::Error& e) {
            fmt::print("Encountered Halide error in forward pipeline of {}:\n{}\n", funcName, e.what());
            throw;
        }
        auto trial = HalideGen::BufferAdaptor<float>(forwardTrialResult);

        // Initialize output grad buffer.
        auto outputGradBuffer = Halide::Buffer<float>(outputBufferShape);
        if constexpr (DoInitialization) {
            auto outputGradProxy = HalideGen::BufferRefAdaptor<float>(outputGradBuffer);
            outputGradBuffer.for_each_element(ReverseArguments(std::bind_front(std::forward<decltype(outputGradInitializer)>(outputGradInitializer), std::ref(outputGradProxy))));
        }
        backwardInputs.back().set(outputGradBuffer);

        // Compute the backward result.
        backwardPipeline.compile_jit(target);
        auto realizationArgs = [&]<std::size_t... i>(std::index_sequence<i...>) {
            return Halide::Pipeline::RealizationArg(inputGradsBuffers.at(i)...);
        }(std::make_index_sequence<sizeof...(InputInitializers)>());
        try {
            backwardPipeline.realize(std::move(realizationArgs), target);
        } catch (const Halide::Error& e) {
            fmt::print("Encountered Halide error in backward pipeline of {}:\n{}\n", funcName, e.what());
            throw;
        }
        std::vector<HalideGen::BufferAdaptor<float>> backwardTrials;
        for (auto& inputGradBuffer: inputGradsBuffers) {
            backwardTrials.emplace_back(std::move(inputGradBuffer));
            backwardTrials.back().content.copy_to_host();
        }

        return { std::move(pipeline), std::move(trial), std::move(backwardPipeline), std::move(backwardTrials) };
    }

};

} // namespace kas
