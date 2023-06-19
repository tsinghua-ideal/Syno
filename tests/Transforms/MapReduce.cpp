#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, map_reduce) {
    MapReduce MapReduce { 0, sizeH * sizeW, MapReduce::MapType::Identity, MapReduce::ReduceType::Sum };
    Interface in { &MapReduce, dimH, dimW, dimCH };
    auto tensorView = TensorView { in };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H*W,H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            float temp_ri_0 = 0;
            for (int ri_0 = 0; ri_0 < H*W; ri_0++) {
                temp_ri_0 += in_0[ri_0,i_0,i_1,i_2];
            }
            out[i_0,i_1,i_2] = temp_ri_0;
        }
    }
}
)");
    auto [_0, _1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "map_reduce", false, false,
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 32 * i + 8 * j + k;
        },
        [](auto&& buf, int i, int j, int k, int l) {
            buf(i, j, k, l) = 32 * j + 8 * k + l + i;
        }
    );
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 16 * (32 * i + 8 * j + k) + 120);
            }
        }
    }
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 8; l++) {
                    ASSERT_EQ(derivatives[0](i, j, k, l), 32 * j + 8 * k + l);
                }
            }
        }
    }
}

} // namespace kas
