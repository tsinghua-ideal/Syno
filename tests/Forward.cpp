#include <fmt/core.h>
#include <gtest/gtest.h>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms/Forward.hpp"

using namespace kas;


TEST(forward_tests, pooling) {
    using SizeName = BindingContext::Metadata;
    BindingContext ctx { std::vector<SizeName> { SizeName("N", 64), SizeName("H", 128), SizeName("W", 128) }, std::vector<SizeName> { SizeName("C", 3), SizeName("K", 5) } };
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

    auto dimH_dot_K_and_dimW_dot_K_reduce = dimH_dot_K_and_dimW_dot_K.reduce(0, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum);
    // [N, C, H/K, W/K], where the K^2 is reduced.

    auto dimN_input = dimN.input(0);
    auto dimC_input = dimC.input(1);
    auto dimH_over_K_input = dimH_over_K.input(2);
    auto dimW_over_K_input = dimW_over_K.input(3);

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
}