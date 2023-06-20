#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stack>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>
#include "Halide.h"

#include "KAS/CodeGen/Common.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Functional.hpp"
#include "KAS/Utils/Tuple.hpp"


namespace kas {

inline Halide::Expr ConcretizeSize(const ConcreteConsts& consts, const Size& value) {
    return value.eval<int>(consts);
}

template<SizeRange R>
Halide::Region ConcretizeShape(const ConcreteConsts& consts, R&& shape, bool reverse = true) {
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
    struct Constraint {
        Halide::Expr inequality;
        bool likely;
    };

    int position; // Still, -1 means output tensor.
    std::vector<Halide::Var> outerLoops; // In reverse order.
    std::vector<Halide::Var> rfactoredInnerLoops; // In priority order.
    Halide::RDom rfactoredDomain;
    Halide::RDom reductionDomain;
    std::vector<Constraint> constraints;
    std::vector<std::vector<Halide::Expr>> inputs; // Inner arrays are in reverse order.
    std::vector<Halide::Expr> output; // In reverse order.

    // Description of the expression.
    mutable TensorExpression expression;
    std::optional<Halide::Expr> divBy;
};

class HalideGen {
public:
    struct Options {
        enum class AutoScheduler {
            ComputeRoot, Mullapudi2016, Li2018, Adams2019, Anderson2021,
        };
        bool useGPU = true;
        AutoScheduler scheduler = AutoScheduler::Li2018;
        std::size_t rfactorThreshold = 32;
        float inBoundsLikelyThreshold = 0.3f;
    };

private:
    const BindingContext& ctx;
    const TensorView& tensorView;
    Options options;

public:
    static void GuardAutoSchedulers(); // Load the Halide auto scheduler plugins.

    HalideAccess lowerToAccess(const ConcreteConsts& consts, const AbstractAccess& access) const;

    // If lowering a backward pipeline, the output tensor is the gradient of the output tensor.
    Halide::Func lowerAccessToFunc(std::vector<Halide::ImageParam>& inputTensors, Halide::ImageParam *outputTensor, const HalideAccess& access, std::string_view funcName) const;

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
    ForwardArgsAndFunc createFunc(const ConcreteConsts& consts, const ConcreteShapes& shapes, std::string_view funcName) const;

    using BackwardArgsAndFuncs = std::pair<std::vector<Halide::ImageParam>, std::vector<Halide::Func>>;
    // Returns the (input (including the gradient of output), gradient funcs) pair.
    BackwardArgsAndFuncs createFuncGrad(const ConcreteConsts& consts, const ConcreteShapes& shapes, std::string_view funcName) const;

    using ForwardAndBackwardFuncs = std::tuple<std::vector<Halide::ImageParam>, Halide::Func, std::vector<Halide::ImageParam>, std::vector<Halide::Func>>;
    // Returns the forward and backward funcs.
    ForwardAndBackwardFuncs createPipelines(const ConcreteConsts& consts, std::string_view funcName) const;

    // Applys auto-schedulers on the funcs.
    using ScheduledPipelins = std::pair<Halide::Pipeline, Halide::Pipeline>;
    static ScheduledPipelins ApplyAutoScheduler(Halide::Func& forwardFunc, std::vector<Halide::Func>& backwardFuncs, const Halide::Target& target, Options::AutoScheduler scheduler, bool verbose);

    static void GenerateFromPipelines(std::vector<Halide::ImageParam>& forwardInputs, std::vector<Halide::ImageParam>& backwardInputs, Halide::Pipeline& forwardPipeline, Halide::Pipeline& backwardPipeline, std::filesystem::path outputPath, std::string_view funcName, const Halide::Target& target);

    HalideGen(const BindingContext& ctx, const TensorView& tensorView, Options options):
        ctx { ctx },
        tensorView { tensorView },
        options { std::move(options) }
    {}

    void generate(std::filesystem::path outputDir, std::string_view funcName, const ConcreteConsts& consts) const;

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

    template<typename F>
    static auto ReverseIntArgs(F&& f) {
        return [&]<typename... Args, auto = std::conjunction_v<std::is_integral<std::decay_t<Args>>...>, typename = decltype(f(std::declval<Args>()...))>(Args... args) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return f(std::get<sizeof...(Is) - 1 - Is>(std::tie(args...))...);
            }(std::index_sequence_for<decltype(args)...>{});
        };
    }

    // For test use only.
    struct Realization {
        PaddedConsts paddedConsts;
        Halide::Pipeline pipeline;
        HalideGen::BufferAdaptor<float> trial;
        Halide::Pipeline backwardPipeline;
        std::vector<HalideGen::BufferAdaptor<float>> backwardTrials;
    };
    template<bool DoInitialization = true, typename... InputInitializers>
    Realization performTrial(const std::map<std::string, std::size_t>& mappings, auto&& funcName, bool createStaticLibrary, bool verbose, auto&& outputGradInitializer, InputInitializers&&... inputInitializers) const {
        auto unpaddedConsts = ctx.realizeConsts(mappings);
        auto consts = tensorView.computePadding(ctx, unpaddedConsts);
        auto shapes = concretizeShapes(consts);
        auto [inputs, func, backwardInputs, backwardFuncs] = createPipelines(consts, std::forward<decltype(funcName)>(funcName));

        std::vector<Halide::Buffer<float>> inputBuffers;
        std::vector<Halide::Buffer<float>> inputGradsBuffers;
        if constexpr (DoInitialization) {
            // Initialize input buffers.
            KAS_ASSERT(tensorView.getUnderlyingTensors().size() == sizeof...(inputInitializers));
            std::tuple<decltype(inputInitializers)...> initializerTuple = { std::forward<decltype(inputInitializers)>(inputInitializers)... };
            auto setter = [&]<std::size_t i>() {
                std::vector<int> inputBufferShape = ConcreteShapes::RegionToVector(shapes.inputShapes.at(i));
                inputGradsBuffers.emplace_back(inputBufferShape);
                inputBuffers.emplace_back(inputBufferShape);
                auto& inputBuffer = inputBuffers.back();
                if constexpr (DoInitialization) {
                    auto proxy = HalideGen::BufferRefAdaptor<float>(inputBuffer);
                    inputBuffer.for_each_element(ReverseIntArgs(std::bind_front(std::get<i>(initializerTuple), std::ref(proxy))));
                }
                inputs.at(i).set(inputBuffer);
                backwardInputs.at(i).set(inputBuffer);
            };
            [&]<std::size_t... i>(std::index_sequence<i...>) {
                (setter.template operator()<i>(), ...);
            }(std::make_index_sequence<sizeof...(InputInitializers)>());
        } else {
            for (std::size_t i = 0; i < tensorView.getUnderlyingTensors().size(); ++i) {
                const auto& inputShape = shapes.inputShapes.at(i);
                std::vector<int> inputBufferShape = ConcreteShapes::RegionToVector(inputShape);
                inputGradsBuffers.emplace_back(inputBufferShape);
                inputBuffers.emplace_back(inputBufferShape);
                inputs.at(i).set(inputBuffers.back());
                backwardInputs.at(i).set(inputBuffers.back());
            }
        }

        // Compute the forward result.
        auto target = GetHostTarget(options.useGPU, true);
        auto [pipeline, backwardPipeline] = HalideGen::ApplyAutoScheduler(func, backwardFuncs, target, options.scheduler, verbose);

        if (createStaticLibrary) {
            std::string outputDir = fmt::format("./kernel_{}", funcName);
            std::string indexedFuncName = fmt::format("{}_0", funcName);
            HalideGen::GenerateFromPipelines(inputs, backwardInputs, pipeline, backwardPipeline, outputDir, indexedFuncName, target);
            int err = LinkObjects(outputDir, fmt::format("{}.so", funcName), { fmt::format("{}.o", indexedFuncName), fmt::format("{}_grad.o", indexedFuncName) });
            KAS_ASSERT(err == 0, "Failed to link objects.");
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
            outputGradBuffer.for_each_element(ReverseIntArgs(std::bind_front(std::forward<decltype(outputGradInitializer)>(outputGradInitializer), std::ref(outputGradProxy))));
        }
        backwardInputs.back().set(outputGradBuffer);

        // Compute the backward result.
        backwardPipeline.compile_jit(target);
        auto realizationArgs = [&]() -> Halide::Pipeline::RealizationArg {
            if constexpr (DoInitialization) {
                return [&]<std::size_t... i>(std::index_sequence<i...>) {
                    return Halide::Pipeline::RealizationArg(inputGradsBuffers.at(i)...);
                }(std::make_index_sequence<sizeof...(InputInitializers)>());
            } else {
                switch (tensorView.getUnderlyingTensors().size()) {
                case 1: return Halide::Pipeline::RealizationArg(inputGradsBuffers[0]);
                case 2: return Halide::Pipeline::RealizationArg(inputGradsBuffers[0], inputGradsBuffers[1]);
                case 3: return Halide::Pipeline::RealizationArg(inputGradsBuffers[0], inputGradsBuffers[1], inputGradsBuffers[2]);
                case 4: return Halide::Pipeline::RealizationArg(inputGradsBuffers[0], inputGradsBuffers[1], inputGradsBuffers[2], inputGradsBuffers[3]);
                case 5: return Halide::Pipeline::RealizationArg(inputGradsBuffers[0], inputGradsBuffers[1], inputGradsBuffers[2], inputGradsBuffers[3], inputGradsBuffers[4]);
                default: KAS_CRITICAL("Unsupported number of input tensors: {}", tensorView.getUnderlyingTensors().size());
                }
            }
        }();
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

        return { { unpaddedConsts, consts }, std::move(pipeline), std::move(trial), std::move(backwardPipeline), std::move(backwardTrials) };
    }

};

} // namespace kas
