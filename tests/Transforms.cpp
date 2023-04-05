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
    auto [shareOpL, shareOpR] = shareOp.getInputs();
    Interface in { shareOpL, shareOpR, dimW, dimCH };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H,H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[i_0,i_0,i_1,i_2];
        }
    }
}
)");
    auto [_1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "share", false,
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 5 * i;
        },
        [](auto&& buf, int i, int j, int k, int l) {
            buf(i, j, k, l) = 4 * i + j;
        }
    );
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 5 * i);
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 8; l++) {
                    ASSERT_EQ(derivatives[0](i, j, k, l), static_cast<float>(i == j) * (4 * i + j));
                }
            }
        }
    }
}

TEST_F(transforms_tests, map_reduce) {
    MapReduceOp mapReduceOp { 0, sizeH * sizeW, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum };
    Interface in { &mapReduceOp, dimH, dimW, dimCH };
    auto tensorView = TensorView { { in } };
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
    auto [_1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "map_reduce", false,
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
    auto tensorView = TensorView { { in } };
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
    auto [_1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "shift", false,
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

TEST_F(transforms_tests, stride) {
    StrideOp strideOp { dimH, sizeC };
    Interface in { strideOp.getInput(), dimW, dimCH };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[c*H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[(c)*(i_0),i_1,i_2];
        }
    }
}
)");
    auto [_1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "stride", false,
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 32 * 2 * i + 8 * j + k;
        },
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 32 * i + 8 * j + k;
        }
    );
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 32 * 2 * i + 8 * j + k);
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                if (i % 2 == 0) {
                    ASSERT_EQ(derivatives[0](i, j, k), 32 * i + 8 * j + k);
                } else {
                    ASSERT_EQ(derivatives[0](i, j, k), 0);
                }
            }
        }
    }
}

TEST_F(transforms_tests, unfold) {
    auto itC = Iterator { 2, sizeC };
    UnfoldOp unfoldOp { dimH, &itC };
    Interface in { unfoldOp.getInput(), dimW };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[H,W,c]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H,W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, AbstractAccess::Output),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c; i_2++) {
            out[i_0,i_1,i_2] = in_0[restrict(((i_0)+(i_2))-(((c)-(1))/(2)),0,H),i_1];
        }
    }
}
)");
    auto [_1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 3}},
        "unfold", false,
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 12 * j + 3 * i + k;
        },
        [](auto&& buf, int i, int j) {
            buf(i, j) = 4 * i + j;
        }
    );
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                int access = i + k - 1;
                // We now use replicate padding.
                access = std::min(std::max(access, 0), 4 - 1);
                ASSERT_EQ(outputBuffer(i, j, k), 4 * access + j);
            }
        }
    }
    // Since we use replicate padding, handle with care. On the boundary, the elements are accumulated for additional times.
    // We just simulate this to simplify things.
    std::array<std::array<int, 4>, 4> d {};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                int access = i + k - 1;
                access = std::min(std::max(access, 0), 4 - 1);
                d[access][j] += 12 * j + 3 * i + k;
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            ASSERT_EQ(derivatives[0](i, j), d[i][j]);
        }
    }
}

TEST_F(transforms_tests, merge) {
    MergeOp mergeOp { dimCH, sizeH };
    auto [mergeOpL, mergeOpR] = mergeOp.getInputs();
    Interface in { mergeOpL, mergeOpR, dimH, dimW };
    auto tensorView = TensorView { { in } };
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
    auto [_1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "merge", false,
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
    auto tensorView = TensorView { { in } };
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
    auto [_1, outputBuffer, _2, derivatives] = HalideGen(ctx, tensorView, {}).performTrial(
        {{"H", 4}, {"W", 4}, {"c", 2}},
        "split", false,
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
    auto [sL, sR] = store.get<ShareOp>(dimH)->getInputs();
    ASSERT_NE(sL, sR);
    ASSERT_NE(s1, sL);
    ASSERT_NE(s1, sR);
}

} // namespace kas
