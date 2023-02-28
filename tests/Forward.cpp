#include <chrono>

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <Halide.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms/Forward.hpp"


namespace kas {

TEST(forward_tests, pooling) {
    constexpr int n = 64, c = 3, h = 128, w = 128, k = 5;

    using SizeName = BindingContext::Metadata;
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

    auto gen = HalideGen(ctx, tensorView, {
        .useGPU = false,
        .scheduler = HalideGen::Options::AutoScheduler::Adams2019,
        .zeroPadding = false,
    });
    std::map<std::string, std::size_t> mappings {{"N", n}, {"H", h}, {"W", w}, {"C", c}, {"K", k}};
    auto [inputs, func, _0, _1] = gen.createPipelines(mappings, "pooling");
    KAS_ASSERT(inputs.size() == 1);
    auto& input = inputs[0];
    // Give special care to the column-major layout.
    auto inputBuffer = Halide::Buffer<float, 4>(std::vector<int> { w, h, c, n});
    inputBuffer.for_each_element([&](int l, int k, int j, int i) {
        inputBuffer(l, k, j, i) = static_cast<float>(i + j + k + l);
    });
    input.set(inputBuffer);
    Halide::Pipeline p = func;
    HalideGen::GuardAutoSchedulers();
    p.apply_autoscheduler(Halide::get_host_target(), Halide::AutoschedulerParams { "Adams2019" });
    p.print_loop_nest();
    p.compile_jit();

    constexpr int x = 100;
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < x; ++i) {
        p.realize(std::vector<int> { w / k, h / k, c, n });
    }
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    fmt::print("Pooling x{}: {} ms.\n", x, duration);

    gen.generate("./kernel_pooling", "pooling", mappings);
}

TEST(forward_tests, conv2d) {
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

    HalideGen gen { ctx, tensorView, {
        .useGPU = false,
        .scheduler = HalideGen::Options::AutoScheduler::Adams2019,
        .zeroPadding = false,
    } };
    std::map<std::string, std::size_t> mappings {{"N", n}, {"H", h}, {"W", w}, {"C_in", c_in}, {"C_out", c_out}, {"K", k}};
    gen.generate("./kernel_conv2d", "conv2d", mappings);
}

} // namespace kas
