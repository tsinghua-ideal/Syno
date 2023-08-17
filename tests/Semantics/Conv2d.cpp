#include "Prelude.hpp"


namespace kas {

TEST_F(semantics_tests, conv2d) {
    constexpr int n = 100, c_in = 64, c_out = 96, h = 16, w = 16, k = 3;

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
    BindingContext::DebugPublicCtx = &ctx;
    Forward::Factory factory { ctx };
    auto [sizeN, sizeCin, sizeCout, sizeH, sizeW, sizeK] = factory.getSizes("N", "C_in", "C_out", "H", "W", "K");

    auto [dimN, dimCin_input, dimH, dimW] = factory.makeDimsOfSizes(sizeN, sizeCin, sizeH, sizeW);
    // [N, C_in, H, W], the input.

    auto [dimCout, dimCin_filter, dimK1, dimK2] = factory.makeDimsOfSizes(sizeCout, sizeCin, sizeK, sizeK);
    // [C_out, C_in, K, K], the filter.

    // The input tensors are blended into [N, C_in, H, W, C_out, C_in, K, K].

    auto [dimH_over_K, dimH_dot_K] = Forward::UnfoldOp::Create(dimH, sizeK);
    auto [dimW_over_K, dimW_dot_K] = Forward::UnfoldOp::Create(dimW, sizeK);
    // [N, C_in, H, K, W, K, C_out, C_in, K, K], where H and W are unfolded.

    auto dimCin_shared = Forward::ShareOp::Create(dimCin_input, dimCin_filter);
    auto dimK1_shared = Forward::ShareOp::Create(dimH_dot_K, dimK1);
    auto dimK2_shared = Forward::ShareOp::Create(dimW_dot_K, dimK2);
    // [N, C_in, H, W, C_out, K, K], where C_in, K1, and K2 are shared.

    dimN.output(0);
    dimCout.output(1);
    dimH_over_K.output(2);
    dimW_over_K.output(3);
    dimK2_shared.reduce(0, Reduce::MapType::Identity, Reduce::ReduceType::Sum);
    dimK1_shared.reduce(1, Reduce::MapType::Identity, Reduce::ReduceType::Sum);
    dimCin_shared.reduce(2, Reduce::MapType::Identity, Reduce::ReduceType::Sum);
    // [N, C_out, H, W], the output.

    // TODO: Add bias.
    auto tensorView = TensorView({
        {{dimN, dimCin_input, dimH, dimW}, {}},
        {{dimCout, dimCin_filter, dimK1, dimK2}, {}}
    }, Parser("in_0 * in_1").parseTensorExpression());
    ASSERT_EQ("[C_out, C_in, K, K]", tensorView.getUnderlyingTensors()[1].shapeToString(ctx));
    ASSERT_EQ(tensorView.printNestedLoops(ctx, TensorExpression::Output),
R"(for (int i_0 = 0; i_0 < N; i_0++) {
    for (int i_1 = 0; i_1 < C_out; i_1++) {
        for (int i_2 = 0; i_2 < H; i_2++) {
            for (int i_3 = 0; i_3 < W; i_3++) {
                float temp_ri_2 = 0;
                for (int ri_2 = 0; ri_2 < K; ri_2++) {
                    float temp_ri_1 = 0;
                    for (int ri_1 = 0; ri_1 < K; ri_1++) {
                        float temp_ri_0 = 0;
                        for (int ri_0 = 0; ri_0 < C_in; ri_0++) {
                            temp_ri_0 += in_0[i_0, ri_0, restrict((i_2 + ri_2) - (K) / 2, 0, H), restrict((i_3 + ri_1) - (K) / 2, 0, W)] * in_1[i_1, ri_0, ri_2, ri_1];
                        }
                        temp_ri_1 += temp_ri_0;
                    }
                    temp_ri_2 += temp_ri_1;
                }
                out[i_0, i_1, i_2, i_3] = temp_ri_2;
            }
        }
    }
}
)");
    fmt::print("Gradient for input:\n{}", tensorView.printNestedLoops(ctx, TensorExpression::Input<0>));
    fmt::print("Gradient for weight:\n{}", tensorView.printNestedLoops(ctx, TensorExpression::Input<1>));
    auto subgraphs = tensorView.getSubgraphs();
    ASSERT_EQ(subgraphs.outputTensor.toString(ctx), "(([N, C_in, H, W] -> [N, C_in, K, H, K, W]), [C_out, C_in, K, K] -> [N, C_out, H, W])");

    auto funcName = "conv2d";
    auto gvGen = GraphvizGen { tensorView, ctx };
    gvGen.generate(fmt::format("./kernel_{0}/{0}.dot", funcName), funcName);
    auto gen = HalideGen { ctx, tensorView, options };
    auto mappings = Mappings {{"N", n}, {"H", h}, {"W", w}, {"C_in", c_in}, {"C_out", c_out}, {"K", k}};
    auto pytorchGen = PyTorchGen { ctx, tensorView };
    {
        std::ofstream file(fmt::format("./kernel_{0}/{0}.py", funcName));
        pytorchGen.generatePrelude(file);
        pytorchGen.generate(file, funcName, tensorView.getForwardAccess(), tensorView.computeConsts(ctx, mappings));
    }
    auto in_0 = new float[n][c_in][h][w]();
    auto in_1 = new float[c_out][c_in][k][k]();
    auto out_grad = new float[n][c_out][h][w]();
    auto [consts, pipeline, trial, backwardPipeline, backwardTrials] = gen.performTrial(mappings, funcName, createStaticLibrary, true,
        [&](auto&& grad, int N, int C_out, int H, int W) {
            float res = random();
            grad(N, C_out, H, W) = res;
            out_grad[N][C_out][H][W] = res;
        },
        [&](auto&& inputBuffer, int N, int C_in, int H, int W) {
            float res = random();
            inputBuffer(N, C_in, H, W) = res;
            in_0[N][C_in][H][W] = res;
        },
        [&](auto&& weightBuffer, int C_out, int C_in, int K1, int K2) {
            float res = random();
            weightBuffer(C_out, C_in, K1, K2) = res;
            in_1[C_out][C_in][K1][K2] = res;
        }
    );
    fmt::print("Consts: {}\n", consts.toString(ctx));

    bool success = true;
    if (doSemanticTests) {
        fmt::print("Running semantic tests for {}...\n", funcName);
        auto in_0_grad = new float[n][c_in][h][w]();
        auto in_1_grad = new float[c_out][c_in][k][k]();
        constexpr float eps = 1e-4;
        std::atomic<std::int64_t> cntCorrect = 0, cntIncorrect = 0;
        #ifndef __clang__ // This is weird: clangd crashes upon this pragma.
        #pragma omp parallel for reduction(+:in_1_grad[:c_out][:c_in][:k][:k])
        #endif
        for (int N = 0; N < n; ++N) {
            for (int C_out = 0; C_out < c_out; ++C_out) {
                for (int H = 0; H < h; ++H) {
                    for (int W = 0; W < w; ++W) {
                        float sum = 0;
                        for (int C_in = 0; C_in < c_in; ++C_in) {
                            for (int K1 = 0; K1 < k; ++K1) {
                                for (int K2 = 0; K2 < k; ++K2) {
                                    auto restrictH = H + K1 - (k - 1) / 2;
                                    auto restrictW = W + K2 - (k - 1) / 2;
                                    if (0 <= restrictH && restrictH < h && 0 <= restrictW && restrictW < w) {
                                        sum += in_0[N][C_in][restrictH][restrictW] * in_1[C_out][C_in][K1][K2];
                                        in_0_grad[N][C_in][restrictH][restrictW] += out_grad[N][C_out][H][W] * in_1[C_out][C_in][K1][K2];
                                        in_1_grad[C_out][C_in][K1][K2] += out_grad[N][C_out][H][W] * in_0[N][C_in][restrictH][restrictW];
                                    }
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
            fmt::print("Output tensor: N = {} done. Total correct = {}, incorrect = {}\n", N, cntCorrect.load(), cntIncorrect.load());
        }
        fmt::print("Output tensor: Total correct = {}, incorrect = {}\n", cntCorrect.load(), cntIncorrect.load());
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
            fmt::print("Input tensor: N = {} done. Total correct = {}, incorrect = {}\n", N, cntCorrect.load(), cntIncorrect.load());
        }
        fmt::print("Input tensor: Total correct = {}, incorrect = {}\n", cntCorrect.load(), cntIncorrect.load());
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
        fmt::print("Weight tensor: Total correct = {}, incorrect = {}\n", cntCorrect.load(), cntIncorrect.load());
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
