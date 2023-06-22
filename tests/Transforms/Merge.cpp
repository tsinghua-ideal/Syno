#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, merge) {
    MergeOp mergeOp { dimCH, sizeH };
    std::vector<Dimension> in { mergeOp.getInputL(), mergeOp.getInputR(), dimH, dimW };
    auto tensorView = TensorView({ in }, TensorExpression::ProductOfTensors(1));
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H, W, c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[c, H, H, W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, TensorExpression::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0, i_1, i_2] = in_0[i_2 / (H), i_2 % H, i_0, i_1];
        }
    }
}
)");
    auto [_0, _1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "merge", false, false,
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 32 * i + 8 * j + k;
        },
        [](auto&& buf, int i, int j, int k, int l) {
            buf(i, j, k, l) = 4 * i + j;
        }
    );
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), k);
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 4; l++) {
                    ASSERT_EQ(derivatives[0](i, j, k, l), 32 * k + 8 * l + 4 * i + j);
                }
            }
        }
    }
}

} // namespace kas
