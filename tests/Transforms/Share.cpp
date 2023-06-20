#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, share) {
    ShareOp shareOp { dimH };
    Interface in0 { shareOp.getInputL(), dimW, dimCH };
    Interface in1 { shareOp.getInputR() };
    auto tensorView = TensorView { in0, in1 };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[1].shapeToString(ctx), "[H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, TensorExpression::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[i_0,i_1,i_2] * in_1[i_0];
        }
    }
}
)");
    auto [_0, _1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "share", false, false,
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 2 * i;
        },
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 3 * i;
        },
        [](auto&& buf, int i) {
            buf(i) = 5 * i;
        }
    );
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 15 * i * i);
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(derivatives[0](i, j, k), 10 * i * i);
            }
        }
        ASSERT_EQ(derivatives[1](i), (4 * 8) * 6 * i * i);
    }
}

} // namespace kas
