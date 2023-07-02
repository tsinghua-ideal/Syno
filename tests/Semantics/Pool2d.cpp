#include "Prelude.hpp"


namespace kas {

TEST_F(semantics_tests, pool2d) {
    constexpr int n = 64, c = 3, h = 128, hp = 130, w = 128, wp = 130, k = 5;

    BindingContext ctx { std::vector<SizeName> {
        SizeName { .alias = "N", .estimate = n },
        SizeName { .alias = "H", .estimate = h },
        SizeName { .alias = "W", .estimate = w },
    }, std::vector<SizeName> {
        SizeName { .alias = "C", .estimate = c },
        SizeName { .alias = "K", .estimate = k },
    } };
    Forward::Factory factory { ctx };
    auto [sizeN, sizeC, sizeH, sizeW, sizeK] = factory.getSizes("N", "C", "H", "W", "K");

    auto [dimN, dimC, dimH, dimW] = factory.makeDimsOfSizes(sizeN, sizeC, sizeH, sizeW);
    // [N, C, H, W], the input.

    auto [dimH_over_K, dimH_dot_K] = Forward::SplitOp::Create(dimH, sizeK);
    // [N, C, H/K, K, W], where H is split into H/K and K.

    auto [dimW_over_K, dimW_dot_K] = Forward::SplitOp::Create(dimW, sizeK);
    // [N, C, H/K, K, W/K, K], where W is split into W/K and K.

    auto dimH_dot_K_and_dimW_dot_K = Forward::MergeOp::Create(dimH_dot_K, dimW_dot_K);
    // [N, C, H/K, W/K, K^2], where the two K from H and W are merged into K^2.

    dimH_dot_K_and_dimW_dot_K.reduce(0, MapReduce::MapType::Identity, MapReduce::ReduceType::Mean);
    // [N, C, H/K, W/K], where the K^2 is reduced.

    dimN.output(0);
    dimC.output(1);
    dimH_over_K.output(2);
    dimW_over_K.output(3);

    std::vector<Dimension> in { dimN, dimC, dimH, dimW };
    auto tensorView = TensorView({ in }, Parser("in_0").parseTensorExpression());
    ASSERT_EQ(tensorView.printNestedLoops(ctx, TensorExpression::Output),
R"(for (int i_0 = 0; i_0 < N; i_0++) {
    for (int i_1 = 0; i_1 < C; i_1++) {
        for (int i_2 = 0; i_2 < K^-1*H; i_2++) {
            for (int i_3 = 0; i_3 < K^-1*W; i_3++) {
                float temp_ri_0 = 0;
                for (int ri_0 = 0; ri_0 < K^2; ri_0++) {
                    temp_ri_0 += (in_0[i_0, i_1, i_2 * K + ri_0 / (K), i_3 * K + ri_0 % K]) / (K^2);
                }
                out[i_0, i_1, i_2, i_3] = temp_ri_0;
            }
        }
    }
}
)");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, TensorExpression::Input<0>),
R"(for (int i_0 = 0; i_0 < N; i_0++) {
    for (int i_1 = 0; i_1 < C; i_1++) {
        for (int i_2 = 0; i_2 < H; i_2++) {
            for (int i_3 = 0; i_3 < W; i_3++) {
                grad_in_0[i_0, i_1, i_2, i_3] = (grad_out[i_0, i_1, i_2 / (K), i_3 / (K)]) / (K^2);
            }
        }
    }
}
)");

    auto funcName = "pool2d";
    auto gvGen = GraphvizGen { tensorView, ctx };
    gvGen.generate("./kernel_" + std::string(funcName) + "/" + std::string(funcName) + ".dot", funcName);
    auto gen = HalideGen { ctx, tensorView, options };
    auto mappings = Mappings {{"N", n}, {"H", h}, {"W", w}, {"C", c}, {"K", k}};
    auto [consts, pipeline, trial, backwardPipeline, backwardTrials] = gen.performTrial(mappings, funcName, createStaticLibrary, true,
        [](auto&& grad, int i, int j, int k, int l) {
            grad(i, j, k, l) = static_cast<float>(i + j + k + l);
        },
        [](auto&& inputBuffer, int i, int j, int k, int l) {
            inputBuffer(i, j, k, l) = static_cast<float>(i + j + k + l);
        }
    );
    fmt::print("Consts: {}\n", consts.toString(ctx));

    fmt::print("Running semantic tests for {}...\n", funcName);
    for (int N = 0; N < n; ++N) {
        for (int C = 0; C < c; ++C) {
            for (int H = 0; H < hp / k; ++H) {
                for (int W = 0; W < wp / k; ++W) {
                    auto res = N + C + k * H + k * W + 2 * (k - 1) / 2;
                    ASSERT_FLOAT_EQ(trial(N, C, H, W), res);
                }
            }
        }
    }
    for (int N = 0; N < n; ++N) {
        for (int C = 0; C < c; ++C) {
            for (int H = 0; H < hp; ++H) {
                for (int W = 0; W < wp; ++W) {
                    bool inBound = H < k * (hp / k) && W < k * (wp / k);
                    if (inBound) {
                        ASSERT_FLOAT_EQ(backwardTrials[0](N, C, H, W) * k * k, N + C + H / k + W / k);
                    } else {
                        ASSERT_FLOAT_EQ(backwardTrials[0](N, C, H, W), 0);
                    }
                }
            }
        }
    }
    fmt::print("{} semantics verified.\n", funcName);

    constexpr int x = 1000;
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < x; ++i) {
        pipeline.realize({wp / k, hp / k, c, n});
    }
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    fmt::print("Pool2d x{}: {} ms.\n", x, duration);
}

} // namespace kas
