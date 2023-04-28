#include <array>
#include <map>

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <Halide.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Functional.hpp"


namespace kas {

// Here, we verify the shape, iterator and codegen semantics of each transform.
class transforms_tests: public ::testing::Test {
protected:
    using SizeName = BindingContext::Metadata;
    BindingContext ctx { std::vector<SizeName> { SizeName("H", 128), SizeName("W", 128) }, std::vector<SizeName> { SizeName("c", 5) } };
    Size sizeH = ctx.getSinglePrimaryVariableSize(0);
    Size sizeW = ctx.getSinglePrimaryVariableSize(1);
    Size sizeC = ctx.getSingleCoefficientVariableSize(0);
    Iterator itH { 0, sizeH }, itW { 1, sizeW }, itCH { 2, sizeC * sizeH };
    Dimension dimH { &itH }, dimW { &itW }, dimCH { &itCH };
};

TEST_F(transforms_tests, share) {
    ShareOp shareOp { dimH };
    Interface in0 { shareOp.getInputL(), dimW, dimCH };
    Interface in1 { shareOp.getInputR() };
    auto tensorView = TensorView { in0, in1 };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[1].shapeToString(ctx), "[H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Output),
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

TEST_F(transforms_tests, map_reduce) {
    MapReduceOp mapReduceOp { 0, sizeH * sizeW, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum };
    Interface in { &mapReduceOp, dimH, dimW, dimCH };
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

TEST_F(transforms_tests, shift) {
    ShiftOp shiftOp { dimH, 1 };
    Interface in { shiftOp.getInput(), dimW, dimCH };
    auto tensorView = TensorView { in };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[((i_0)+(1))%(H),i_1,i_2];
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
            out[i_0,i_1,i_2] = in_0[restrict(((i_0)+((i_2)*(c)))-(((c^2)-(1))/(2)),0,H),i_1];
        }
    }
}
)");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Input<0>),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        float temp_ri_0 = 0;
        for (int ri_0 = 0; ri_0 < c; ri_0++) {
            temp_ri_0 += grad_out[restrict(((i_0)-((ri_0)*(c)))+(((c^2)-(1))/(2)),0,H),i_1,ri_0];
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

TEST_F(transforms_tests, merge) {
    MergeOp mergeOp { dimCH, sizeH };
    Interface in { mergeOp.getInputL(), mergeOp.getInputR(), dimH, dimW };
    auto tensorView = TensorView { in };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[c,H,H,W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[(i_2)/(H),(i_2)%(H),i_0,i_1];
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

TEST_F(transforms_tests, dimension_store) {
    DimensionStore store;
    Dimension s1 = store.get<ShiftOp>(dimH, 1)->getInput();
    Dimension s2 = store.get<ShiftOp>(dimH, 1)->getInput();
    ASSERT_EQ(s1, s2);
    ASSERT_EQ(store.get<ShiftOp>(dimH, 1), store.get<ShiftOp>(dimH, 1));
    Dimension
        sL = store.get<ShareOp>(dimH)->getInputL(),
        sR = store.get<ShareOp>(dimH)->getInputR();
    ASSERT_NE(sL, sR);
    ASSERT_NE(s1, sL);
    ASSERT_NE(s1, sR);
}

} // namespace kas
