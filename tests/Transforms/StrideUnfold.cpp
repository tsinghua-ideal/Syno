#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, stride_unfold) {
    auto itC = Iterator { 2, sizeC }; // C
    StrideOp strideOp { &itC, sizeC }; // C^2
    UnfoldOp unfoldOp { dimH, strideOp.getInput() };
    Interface in { unfoldOp.getInput(), dimW };
    auto tensorView = TensorView { in };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H,W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c; i_2++) {
            out[i_0,i_1,i_2] = in_0[restrict(((i_0)+((i_2)*(c)))-((c^2)/(2)),0,H),i_1];
        }
    }
}
)");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Input<0>),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        float temp_ri_0 = 0;
        for (int ri_0 = 0; ri_0 < c; ri_0++) {
            temp_ri_0 += grad_out[restrict(((i_0)-((ri_0)*(c)))+((c^2)/(2)),0,H),i_1,ri_0];
        }
        grad_in_0[i_0,i_1] = temp_ri_0;
    }
}
)");

    auto [_0, _1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 13}, {"W", 4}, {"c", 3}},
        "unfold", false, false,
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 39 * j + 3 * i + k;
        },
        [](auto&& buf, int i, int j) {
            buf(i, j) = 4 * i + j;
        }
    );
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                int access = i + 3 * k - 9 / 2;
                // We now use zero padding.
                ASSERT_EQ(outputBuffer(i, j, k),
                    (0 <= access && access < 13) ? 4 * access + j : 0
                );
            }
        }
    }
    // We now use zero padding, to make forward and backward pipeline consistent.
    // We just simulate this to simplify things.
    std::array<std::array<int, 4>, 13> d {};
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                int access = i + 3 * k - 9 / 2;
                if (0 <= access && access < 13)
                    d[access][j] += 39 * j + 3 * i + k;
            }
        }
    }
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 4; j++) {
            ASSERT_EQ(derivatives[0](i, j), d[i][j]);
        }
    }
}

} // namespace kas
