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
        .zeroPadding = false,
    };
    const bool doSemanticTests = false;
    const bool createStaticLibrary = true;

    struct Realization {
        Halide::Pipeline pipeline;
        HalideGen::BufferAdaptor<float> trial;
        std::vector<int> outputBufferShape;
    };

    template<typename... InputInitializers>
    Realization getPipeline(HalideGen& gen, const Mappings& mappings, auto&& funcName, InputInitializers&&... inputInitializers) const {
        auto consts = gen.ctx.realizeConsts(mappings);
        auto [inputs, func, backwardInputs, backwardFuncs] = gen.createPipelines(mappings, std::forward<decltype(funcName)>(funcName));
        std::tuple<decltype(inputInitializers)...> initializerTuple = { std::forward<decltype(inputInitializers)>(inputInitializers)... };
        std::vector<Halide::Buffer<float>> inputBuffers;
        auto setter = [&]<std::size_t i>() {
            inputBuffers.emplace_back(gen.getInputBufferShape(consts, i));
            auto& inputBuffer = inputBuffers.back();
            auto proxy = HalideGen::BufferRefAdaptor<float>(inputBuffer);
            inputBuffer.for_each_element(ReverseArguments(std::bind_front(std::get<i>(initializerTuple), std::ref(proxy))));
            inputs.at(i).set(inputBuffer);
        };
        [&]<std::size_t... i>(std::index_sequence<i...>) {
            (setter.template operator()<i>(), ...);
        }(std::make_index_sequence<sizeof...(InputInitializers)>());
        auto target = HalideGen::GetHostTarget(options.useGPU);
        auto [pipeline, backwardPipeline] = HalideGen::ApplyAutoScheduler(func, backwardFuncs, target, options.scheduler, true);
        pipeline.compile_jit(target);
        auto outputBufferShape = gen.getOutputBufferShape(consts);
        auto trial = HalideGen::BufferAdaptor<float>(pipeline.realize(outputBufferShape));
        return { std::move(pipeline), std::move(trial), std::move(outputBufferShape) };
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

    auto i_0 = dimN.input(0);
    auto i_1 = dimC.input(1);
    auto i_2 = dimH_over_K.input(2);
    auto i_3 = dimW_over_K.input(3);

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
    auto [pipeline, trial, outputBufferShape] = getPipeline(gen, mappings, funcName,
        [](auto&& inputBuffer, int i, int j, int k, int l) {
            inputBuffer(i, j, k, l) = static_cast<float>(i + j + k + l);
        }
    );

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
    fmt::print("{} semantics verified.\n", funcName);

    constexpr int x = 1000;
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < x; ++i) {
        pipeline.realize(outputBufferShape);
    }
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    fmt::print("Pooling x{}: {} ms.\n", x, duration);

    if (createStaticLibrary)
        gen.generate("./kernel_pooling", funcName, mappings);
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

    auto i_0 = dimN.input(0);
    auto i_1 = dimCout.input(1);
    auto i_2 = dimH_over_K.input(2);
    auto i_3 = dimW_over_K.input(3);
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
    auto [pipeline, trial, outputBufferShape] = getPipeline(gen, mappings, funcName,
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

    if (doSemanticTests) {
        constexpr float eps = 1e-6;
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
                                }
                            }
                        }
                        if ((trial(N, C_out, H, W) - sum) / sum > eps) {
                            fmt::print("N = {}, C_out = {}, H = {}, W = {} failed. Expected = {}, actual = {}\n", N, C_out, H, W, sum, trial(N, C_out, H, W));
                            ++cntIncorrect;
                        } else {
                            ++cntCorrect;
                        }
                    }
                }
            }
            fmt::print("N = {} done. Total correct = {}, incorrect = {}\n", N, cntCorrect, cntIncorrect);
        }
        fmt::print("{} semantics verified. Correct = {}, incorrect = {}\n", funcName, cntCorrect, cntIncorrect);
    }
    delete[] in_0;
    delete[] in_1;

    if (createStaticLibrary)
        gen.generate("./kernel_conv2d", "conv2d", mappings);
}

} // namespace kas
