#include <chrono>

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <Halide.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/BindingContext.hpp"
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
    auto sizeN = ctx.getSinglePrimaryVariableSize(0);
    auto sizeC = ctx.getSingleCoefficientVariableSize(0);
    auto sizeH = ctx.getSinglePrimaryVariableSize(1);
    auto sizeW = ctx.getSinglePrimaryVariableSize(2);
    auto sizeK = ctx.getSingleCoefficientVariableSize(1);
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

    Interface in { dimN.get(), dimC.get(), dimH.get(), dimW.get() };
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
    auto gen = HalideGen(ctx, tensorView);
    auto consts = gen.realizeConsts({{"N", n}, {"H", h}, {"W", w}, {"C", c}, {"K", k}});
    auto access = gen.evaluateAccess(consts);
    auto [inputs, func] = gen.createFunc(consts, access, "pooling");
    KAS_ASSERT(inputs.size() == 1);
    auto& input = inputs[0];
    // Give special care to the column-major layout.
    auto inputBuffer = Halide::Buffer<float, 4>(std::vector<int> { w, h, c, n});
    inputBuffer.for_each_element([&](int l, int k, int j, int i) {
        inputBuffer(l, k, j, i) = static_cast<float>(i + j + k + l);
    });
    input.set(inputBuffer);
    func.set_estimates({{0, w / k}, {0, h / k}, {0, c}, {0, n}});
    Halide::Pipeline p = func;
    Halide::load_plugin("autoschedule_adams2019");
    try {
    p.apply_autoscheduler(Halide::get_host_target(), Halide::AutoschedulerParams { "Adams2019" });
    p.compile_jit();
    } catch (const Halide::Error& e) {
        fmt::print("Error: {}\n", e.what());
    }
    constexpr int x = 10000;
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < x; ++i) {
        p.realize(std::vector<int> { w / k, h / k, c, n });
    }
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    fmt::print("Pooling x{}: {} ms.\n", x, duration);
}

} // namespace kas
