#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, split) {
    SplitOp splitOp { dimH, dimW };
    Interface in { splitOp.getInput(), dimCH };
    auto tensorView = TensorView { in };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H*W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[((i_0)*(W))+(i_1),i_2];
        }
    }
}
)");
    auto [_0, _1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "split", false, false,
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 4 * i + j + 16 * k;
        },
        [](auto&& buf, int i, int j) {
            buf(i, j) = i + 16 * j;
        }
    );
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 4 * i + j + 16 * k);
            }
        }
    }
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            ASSERT_EQ(derivatives[0](i, j), 16 * j + i);
        }
    }
}

} // namespace kas
