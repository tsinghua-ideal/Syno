#include <chrono>
#include <cstdint>
#include <tuple>

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <Halide.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms/Forward.hpp"
#include "KAS/Utils/Functional.hpp"


namespace kas {

class forward_tests: public ::testing::Test {
protected:
    using SizeName = BindingContext::Metadata;
    using Mappings = std::map<std::string, std::size_t>;
    const HalideGen::Options options = {
        .useGPU = true,
        .scheduler = HalideGen::Options::AutoScheduler::Anderson2021,
    };
    const bool doSemanticTests = true;
    const bool createStaticLibrary = true;

    struct Realization {
        Halide::Pipeline pipeline;
        HalideGen::BufferAdaptor<float> trial;
        std::vector<int> outputBufferShape;
        Halide::Pipeline backwardPipeline;
        std::vector<HalideGen::BufferAdaptor<float>> backwardTrials;
    };

    template<typename... InputInitializers>
    Realization getPipeline(HalideGen& gen, const Mappings& mappings, auto&& funcName, auto&& outputGradInitializer, InputInitializers&&... inputInitializers) const {
        auto consts = gen.ctx.realizeConsts(mappings);
        auto [inputs, func, backwardInputs, backwardFuncs] = gen.createPipelines(mappings, std::forward<decltype(funcName)>(funcName));

        // Initialize input buffers.
        KAS_ASSERT(gen.tensorView.getUnderlyingTensors().size() == sizeof...(inputInitializers));
        std::tuple<decltype(inputInitializers)...> initializerTuple = { std::forward<decltype(inputInitializers)>(inputInitializers)... };
        std::vector<Halide::Buffer<float>> inputBuffers;
        std::vector<Halide::Buffer<float>> inputGradsBuffers;
        auto setter = [&]<std::size_t i>() {
            auto inputBufferShape = gen.getInputBufferShape(consts, i);
            inputGradsBuffers.emplace_back(inputBufferShape);
            inputBuffers.emplace_back(inputBufferShape);
            auto& inputBuffer = inputBuffers.back();
            auto proxy = HalideGen::BufferRefAdaptor<float>(inputBuffer);
            inputBuffer.for_each_element(ReverseArguments(std::bind_front(std::get<i>(initializerTuple), std::ref(proxy))));
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
    
        auto outputBufferShape = gen.getOutputBufferShape(consts);
        auto trial = HalideGen::BufferAdaptor<float>(pipeline.realize(outputBufferShape, target));

        // Initialize output grad buffer.
        auto outputGradBuffer = Halide::Buffer<float>(outputBufferShape);
        auto outputGradProxy = HalideGen::BufferRefAdaptor<float>(outputGradBuffer);
        outputGradBuffer.for_each_element(ReverseArguments(std::bind_front(std::forward<decltype(outputGradInitializer)>(outputGradInitializer), std::ref(outputGradProxy))));
        backwardInputs.back().set(outputGradBuffer);

        // Compute the backward result.
        backwardPipeline.compile_jit(target);
        auto realizationArgs = [&]<std::size_t... i>(std::index_sequence<i...>) {
            return Halide::Pipeline::RealizationArg(inputGradsBuffers.at(i)...);
        }(std::make_index_sequence<sizeof...(InputInitializers)>());
        backwardPipeline.realize(std::move(realizationArgs), target);
        std::vector<HalideGen::BufferAdaptor<float>> backwardTrials;
        for (auto& inputGradBuffer: inputGradsBuffers) {
            backwardTrials.emplace_back(std::move(inputGradBuffer));
            backwardTrials.back().content.copy_to_host();
        }

        return { std::move(pipeline), std::move(trial), std::move(outputBufferShape), std::move(backwardPipeline), std::move(backwardTrials) };
    }
};

TEST_F(forward_tests, pooling) {
    constexpr int n = 64, c = 3, h = 128, w = 128, k = 5;

    BindingContext ctx { std::vector<SizeName> {
        SizeName { .alias = "N", .estimate = n },
        SizeName { .alias = "H", .estimate = h },
        SizeName { .alias = "W", .estimate = w },
    }, std::vector<SizeName> {
        SizeName { .alias = "C", .estimate = c },
        SizeName { .alias = "K", .estimate = k },
    } };
    auto [sizeN, sizeC, sizeH, sizeW, sizeK] = ctx.getSizes("N", "C", "H", "W", "K");
    Forward::Factory factory;

    auto [dimN, dimC, dimH, dimW] = factory.makeSizes(sizeN, sizeC, sizeH, sizeW);
    // [N, C, H, W], the input.

    auto [dimH_over_K, dimH_dot_K] = Forward::SplitOp::Create(dimH, sizeK);
    // [N, C, H/K, K, W], where H is split into H/K and K.

    auto [dimW_over_K, dimW_dot_K] = Forward::SplitOp::Create(dimW, sizeK);
    // [N, C, H/K, K, W/K, K], where W is split into W/K and K.

    auto dimH_dot_K_and_dimW_dot_K = Forward::MergeOp::Create(dimH_dot_K, dimW_dot_K);
    // [N, C, H/K, W/K, K^2], where the two K from H and W are merged into K^2.

    auto ri_0 = dimH_dot_K_and_dimW_dot_K.reduce(0, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
    // [N, C, H/K, W/K], where the K^2 is reduced.

    auto i_0 = dimN.output(0);
    auto i_1 = dimC.output(1);
    auto i_2 = dimH_over_K.output(2);
    auto i_3 = dimW_over_K.output(3);

    Interface in { dimN, dimC, dimH, dimW };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < N; i_0++) {
    for (int i_1 = 0; i_1 < C; i_1++) {
        for (int i_2 = 0; i_2 < K^-1*H; i_2++) {
            for (int i_3 = 0; i_3 < K^-1*W; i_3++) {
                float temp_ri_0 = 0;
                for (int ri_0 = 0; ri_0 < K^2; ri_0++) {
                    temp_ri_0 += in_0[i_0,i_1,((i_2)*(K))+((ri_0)/(K)),((i_3)*(K))+((ri_0)%(K))];
                }
                out[i_0,i_1,i_2,i_3] = temp_ri_0;
            }
        }
    }
}
)");

    auto funcName = "pooling";
    auto gen = HalideGen { ctx, tensorView, options };
    auto mappings = Mappings {{"N", n}, {"H", h}, {"W", w}, {"C", c}, {"K", k}};
    auto [pipeline, trial, outputBufferShape, backwardPipeline, backwardTrials] = getPipeline(gen, mappings, funcName,
        [](auto&& grad, int i, int j, int k, int l) {
            grad(i, j, k, l) = static_cast<float>(i + j + k + l);
        },
        [](auto&& inputBuffer, int i, int j, int k, int l) {
            inputBuffer(i, j, k, l) = static_cast<float>(i + j + k + l);
        }
    );

    fmt::print("Running semantic tests for {}...\n", funcName);
    for (int N = 0; N < n; ++N) {
        for (int C = 0; C < c; ++C) {
            for (int H = 0; H < h / k; ++H) {
                for (int W = 0; W < h / k; ++W) {
                    auto res = k * k * (N + C + k * H + k * W + 2 * (k - 1) / 2);
                    ASSERT_EQ(trial(N, C, H, W), res);
                }
            }
        }
    }
    for (int N = 0; N < n; ++N) {
        for (int C = 0; C < c; ++C) {
            for (int H = 0; H < h; ++H) {
                for (int W = 0; W < w; ++W) {
                    bool inBound = H < k * (h / k) && W < k * (w / k);
                    if (inBound) {
                        ASSERT_EQ(backwardTrials[0](N, C, H, W), N + C + H / k + W / k);
                    } else {
                        ASSERT_EQ(backwardTrials[0](N, C, H, W), 0);
                    }
                }
            }
        }
    }
    fmt::print("{} semantics verified.\n", funcName);

    constexpr int x = 1000;
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < x; ++i) {
        pipeline.realize(outputBufferShape);
    }
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    fmt::print("Pooling x{}: {} ms.\n", x, duration);
}

TEST_F(forward_tests, conv2d) {
    constexpr int n = 64, c_in = 3, c_out = 16, h = 128, w = 128, k = 5;

    using SizeName = BindingContext::Metadata;
    BindingContext ctx { std::vector<SizeName> {
        SizeName { .alias = "N", .estimate = n },
        SizeName { .alias = "H", .estimate = h },
        SizeName { .alias = "W", .estimate = w },
    }, std::vector<SizeName> {
        SizeName { .alias = "C_in", .estimate = c_in },
        SizeName { .alias = "C_out", .estimate = c_out },
        SizeName { .alias = "K", .estimate = k },
    } };
    auto [sizeN, sizeCin, sizeCout, sizeH, sizeW, sizeK] = ctx.getSizes("N", "C_in", "C_out", "H", "W", "K");
    Forward::Factory factory;

    auto [dimN, dimCin_input, dimH, dimW] = factory.makeSizes(sizeN, sizeCin, sizeH, sizeW);
    // [N, C_in, H, W], the input.

    auto [dimCout, dimCin_filter, dimK1, dimK2] = factory.makeSizes(sizeCout, sizeCin, sizeK, sizeK);
    // [C_out, C_in, K, K], the filter.

    // The input tensors are blended into [N, C_in, H, W, C_out, C_in, K, K].

    auto [dimH_over_K, dimH_dot_K] = Forward::UnfoldOp::Create(dimH, sizeK);
    auto [dimW_over_K, dimW_dot_K] = Forward::UnfoldOp::Create(dimW, sizeK);
    // [N, C_in, H, K, W, K, C_out, C_in, K, K], where H and W are unfolded.

    auto dimCin_shared = Forward::ShareOp::Create(dimCin_input, dimCin_filter);
    auto dimK1_shared = Forward::ShareOp::Create(dimK1, dimH_dot_K);
    auto dimK2_shared = Forward::ShareOp::Create(dimK2, dimW_dot_K);
    // [N, C_in, H, W, C_out, K, K], where C_in, K1, and K2 are shared.

    auto i_0 = dimN.output(0);
    auto i_1 = dimCout.output(1);
    auto i_2 = dimH_over_K.output(2);
    auto i_3 = dimW_over_K.output(3);
    auto ri_0 = dimK2_shared.reduce(0, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
    auto ri_1 = dimK1_shared.reduce(1, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
    auto ri_2 = dimCin_shared.reduce(2, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
    // [N, C_out, H, W], the output.

    Interface input { dimN, dimCin_input, dimH, dimW }, weight { dimCout, dimCin_filter, dimK1, dimK2 };
    auto tensorView = TensorView { { input, weight } };
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < N; i_0++) {
    for (int i_1 = 0; i_1 < C_out; i_1++) {
        for (int i_2 = 0; i_2 < H; i_2++) {
            for (int i_3 = 0; i_3 < W; i_3++) {
                float temp_ri_2 = 0;
                for (int ri_2 = 0; ri_2 < C_in; ri_2++) {
                    float temp_ri_1 = 0;
                    for (int ri_1 = 0; ri_1 < K; ri_1++) {
                        float temp_ri_0 = 0;
                        for (int ri_0 = 0; ri_0 < K; ri_0++) {
                            temp_ri_0 += in_0[i_0,ri_2,restrict(((i_2)+(ri_1))-(((K)-(1))/(2)),0,H),restrict(((i_3)+(ri_0))-(((K)-(1))/(2)),0,W)] * in_1[i_1,ri_2,ri_1,ri_0];
                        }
                        temp_ri_1 += temp_ri_0;
                    }
                    temp_ri_2 += temp_ri_1;
                }
                out[i_0,i_1,i_2,i_3] = temp_ri_2;
            }
        }
    }
}
)");

    auto funcName = "conv2d";
    auto gen = HalideGen { ctx, tensorView, options };
    auto mappings = Mappings {{"N", n}, {"H", h}, {"W", w}, {"C_in", c_in}, {"C_out", c_out}, {"K", k}};
    auto in_0 = new std::int64_t[n][c_in][h][w]();
    auto in_1 = new std::int64_t[c_out][c_in][k][k]();
    auto out_grad = new std::int64_t[n][c_out][h][w]();
    auto [pipeline, trial, outputBufferShape, backwardPipeline, backwardTrials] = getPipeline(gen, mappings, funcName,
        [&](auto&& grad, int N, int C_out, int H, int W) {
            std::int64_t res = W + w * (H + h * (C_out + c_out * static_cast<std::int64_t>(N)));
            grad(N, C_out, H, W) = static_cast<float>(res);
            out_grad[N][C_out][H][W] = res;
        },
        [&](auto&& inputBuffer, int N, int C_in, int H, int W) {
            std::int64_t res = W + w * (H + h * (C_in + c_in * static_cast<std::int64_t>(N)));
            inputBuffer(N, C_in, H, W) = static_cast<float>(res);
            in_0[N][C_in][H][W] = res;
        },
        [&](auto&& weightBuffer, int C_out, int C_in, int K1, int K2) {
            std::int64_t res = K2 + k * (K1 + k * (C_in + c_in * static_cast<std::int64_t>(C_out)));
            weightBuffer(C_out, C_in, K1, K2) = static_cast<float>(res);
            in_1[C_out][C_in][K1][K2] = res;
        }
    );

    bool success = true;
    if (doSemanticTests) {
        fmt::print("Running semantic tests for {}...\n", funcName);
        auto in_0_grad = new float[n][c_in][h][w]();
        auto in_1_grad = new float[c_out][c_in][k][k]();
        constexpr float eps = 1e-4;
        std::int64_t cntCorrect = 0, cntIncorrect = 0;
        for (int N = 0; N < n; ++N) {
            for (int C_out = 0; C_out < c_out; ++C_out) {
                for (int H = 0; H < h; ++H) {
                    for (int W = 0; W < w; ++W) {
                        float sum = 0;
                        for (int C_in = 0; C_in < c_in; ++C_in) {
                            for (int K1 = 0; K1 < k; ++K1) {
                                for (int K2 = 0; K2 < k; ++K2) {
                                    auto restrictH = std::clamp(H + K1 - (k - 1) / 2, 0, h - 1);
                                    auto restrictW = std::clamp(W + K2 - (k - 1) / 2, 0, w - 1);
                                    sum += in_0[N][C_in][restrictH][restrictW] * in_1[C_out][C_in][K1][K2];
                                    in_0_grad[N][C_in][restrictH][restrictW] += out_grad[N][C_out][H][W] * in_1[C_out][C_in][K1][K2];
                                    in_1_grad[C_out][C_in][K1][K2] += out_grad[N][C_out][H][W] * in_0[N][C_in][restrictH][restrictW];
                                }
                            }
                        }
                        if ((trial(N, C_out, H, W) - sum) / sum > eps) {
                            fmt::print("Output tensor: N = {}, C_out = {}, H = {}, W = {} failed. Expected = {}, actual = {}\n", N, C_out, H, W, sum, trial(N, C_out, H, W));
                            ++cntIncorrect;
                        } else {
                            ++cntCorrect;
                        }
                    }
                }
            }
            fmt::print("Output tensor: N = {} done. Total correct = {}, incorrect = {}\n", N, cntCorrect, cntIncorrect);
        }
        fmt::print("Output tensor: Total correct = {}, incorrect = {}\n", cntCorrect, cntIncorrect);
        if (cntIncorrect > 0) {
            success = false;
        }
        cntCorrect = 0;
        cntIncorrect = 0;
        for (int N = 0; N < n; ++N) {
            for (int C_in = 0; C_in < c_in; ++C_in) {
                for (int H = 0; H < h; ++H) {
                    for (int W = 0; W < w; ++W) {
                        if ((backwardTrials[0](N, C_in, H, W) - in_0_grad[N][C_in][H][W]) / in_0_grad[N][C_in][H][W] > eps) {
                            fmt::print("Input tensor: N = {}, C_in = {}, H = {}, W = {} failed. Expected = {}, actual = {}\n", N, C_in, H, W, in_0_grad[N][C_in][H][W], backwardTrials[0](N, C_in, H, W));
                            ++cntIncorrect;
                        } else {
                            ++cntCorrect;
                        }
                    }
                }
            }
            fmt::print("Input tensor: N = {} done. Total correct = {}, incorrect = {}\n", N, cntCorrect, cntIncorrect);
        }
        fmt::print("Input tensor: Total correct = {}, incorrect = {}\n", cntCorrect, cntIncorrect);
        if (cntIncorrect > 0) {
            success = false;
        }
        cntCorrect = 0;
        cntIncorrect = 0;
        for (int C_out = 0; C_out < c_out; ++C_out) {
            for (int C_in = 0; C_in < c_in; ++C_in) {
                for (int K1 = 0; K1 < k; ++K1) {
                    for (int K2 = 0; K2 < k; ++K2) {
                        if ((backwardTrials[1](C_out, C_in, K1, K2) - in_1_grad[C_out][C_in][K1][K2]) / in_1_grad[C_out][C_in][K1][K2] > eps) {
                            fmt::print("Weight tensor: C_out = {}, C_in = {}, K1 = {}, K2 = {} failed. Expected = {}, actual = {}\n", C_out, C_in, K1, K2, in_1_grad[C_out][C_in][K1][K2], backwardTrials[1](C_out, C_in, K1, K2));
                            ++cntIncorrect;
                        } else {
                            ++cntCorrect;
                        }
                    }
                }
            }
        }
        fmt::print("Weight tensor: Total correct = {}, incorrect = {}\n", cntCorrect, cntIncorrect);
        if (cntIncorrect > 0) {
            success = false;
        }
        fmt::print("{} semantics verification {}\n", funcName, success ? "passed" : "failed");
    }
    delete[] in_0;
    delete[] in_1;
    ASSERT_TRUE(success);
}

} // namespace kas
