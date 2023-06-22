#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, shift) {
    ShiftOp shiftOp { dimH, 1 };
    Interface in { shiftOp.getInput(), dimW, dimCH };
    auto tensorView = TensorView({ in }, TensorExpression::ProductOfTensors(1));
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H, W, c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H, W, c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, TensorExpression::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0, i_1, i_2] = in_0[(i_0 + 1) % H, i_1, i_2];
        }
    }
}
)");
    auto [_0, _1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "shift", false, false,
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 32 * ((i + 1) % 4) + 8 * j + k;
        },
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 32 * i + 8 * j + k;
        }
    );
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 32 * ((i + 1) % 4) + 8 * j + k);
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(derivatives[0](i, j, k), 32 * i + 8 * j + k);
            }
        }
    }
}

} // namespace kas
