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

    template<std::size_t OutputDimensions>
    struct HalideEssentials {
        Halide::ImageParam input;
        Halide::Func func;
        HalideGen::BufferAdaptor<float, OutputDimensions> outputBuffer;
    };

    template<std::size_t InputDimensions, std::size_t OutputDimensions>
    HalideEssentials<OutputDimensions> realize(
        const TensorView& tensorView,
        std::size_t primaryDim, std::size_t coefficientDim,
        const int (&rawInputDimensions)[InputDimensions],
        auto&& inputInitializer,
        const int (&rawOutputDimensions)[OutputDimensions]
    ) const {
        HalideGen gen(ctx, tensorView);
        auto consts = gen.realizeConsts({{"H", primaryDim}, {"W", primaryDim}, {"c", coefficientDim}});
        auto access = gen.evaluateAccess(consts);
        auto [inputs, func] = gen.createFunc(consts, access, "semantic_test");
        KAS_ASSERT(inputs.size() == 1);
        auto& input = inputs[0];
        // Give special care to the column-major layout.
        std::span inputDimensions { rawInputDimensions };
        auto inputBuffer = Halide::Buffer<float, InputDimensions>(std::vector<int>(inputDimensions.rbegin(), inputDimensions.rend()));
        auto proxy = HalideGen::BufferRefAdaptor<float, InputDimensions> { inputBuffer };
        inputBuffer.for_each_element(ReverseArguments<InputDimensions>(std::bind_front(inputInitializer, std::ref(proxy))));
        input.set(inputBuffer);
        func.compute_root();
        std::span outputDimensions { rawOutputDimensions };
        Halide::Buffer<float, OutputDimensions> outputBuffer = func.realize(std::vector<int>(outputDimensions.rbegin(), outputDimensions.rend()), Halide::get_host_target());
        return { std::move(input), std::move(func), { std::move(outputBuffer) } };
    }
};

TEST_F(transforms_tests, share) {
    ShareOp shareOpL { dimH, Order::Left }, shareOpR { dimH, Order::Right };
    Interface in { &shareOpL, &shareOpR, dimW, dimCH };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].accessToString(ctx), "[i_0,i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H,H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[i_0,i_0,i_1,i_2];
        }
    }
}
)");
    auto [input, func, outputBuffer] = realize(
        tensorView, 4, 2,
        {4, 4, 4, 8},
        [](auto&& buf, int i, int j, int k, int l) {
            buf(i, j, k, l) = 4 * i + j;
        },
        {4, 4, 8}
    );
    ASSERT_EQ(input.dimensions(), 4);
    ASSERT_EQ(func.dimensions(), 3);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 5 * i);
            }
        }
    }
}

TEST_F(transforms_tests, map_reduce) {
    MapReduceOp mapReduceOp { 0, sizeH * sizeW, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum };
    Interface in { &mapReduceOp, dimH, dimW, dimCH };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2,ri_0] with ri_0 Sum reduced");
    ASSERT_EQ(tensorView.getShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getReduceShape().toString(ctx), "[H*W]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].accessToString(ctx), "[ri_0,i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H*W,H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
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
    auto [input, func, outputBuffer] = realize(
        tensorView, 4, 2,
        {16, 4, 4, 8},
        [](auto&& buf, int i, int j, int k, int l) {
            buf(i, j, k, l) = 32 * j + 8 * k + l + i;
        },
        {4, 4, 8}
    );
    ASSERT_EQ(input.dimensions(), 4);
    ASSERT_EQ(func.dimensions(), 3);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 16 * (32 * i + 8 * j + k) + 120);
            }
        }
    }
}

TEST_F(transforms_tests, shift) {
    ShiftOp shiftOp { dimH, 1 };
    Interface in { &shiftOp, dimW, dimCH };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].accessToString(ctx), "[((i_0)+(1))%(H),i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[((i_0)+(1))%(H),i_1,i_2];
        }
    }
}
)");
    auto [input, func, outputBuffer] = realize(
        tensorView, 4, 2,
        {4, 4, 8},
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 32 * i + 8 * j + k;
        },
        {4, 4, 8}
    );
    ASSERT_EQ(input.dimensions(), 3);
    ASSERT_EQ(func.dimensions(), 3);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 32 * ((i + 1) % 4) + 8 * j + k);
            }
        }
    }
}

TEST_F(transforms_tests, stride) {
    StrideOp strideOp { dimH, sizeC };
    Interface in { &strideOp, dimW, dimCH };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].accessToString(ctx), "[(c)*(i_0),i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[c*H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[(c)*(i_0),i_1,i_2];
        }
    }
}
)");
    auto [input, func, outputBuffer] = realize(
        tensorView, 4, 2,
        {8, 4, 8},
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = 32 * i + 8 * j + k;
        },
        {4, 4, 8}
    );
    ASSERT_EQ(input.dimensions(), 3);
    ASSERT_EQ(func.dimensions(), 3);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 32 * 2 * i + 8 * j + k);
            }
        }
    }
}

TEST_F(transforms_tests, unfold) {
    auto itC = Iterator { 2, sizeC };
    UnfoldOp unfoldOp { dimH, &itC };
    Interface in { &unfoldOp, dimW };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getShape().toString(ctx), "[H,W,c]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].accessToString(ctx), "[restrict(((i_0)+(i_2))-(((c)-(1))/(2)),0,H),i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H,W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c; i_2++) {
            out[i_0,i_1,i_2] = in_0[restrict(((i_0)+(i_2))-(((c)-(1))/(2)),0,H),i_1];
        }
    }
}
)");
    auto [input, func, outputBuffer] = realize(
        tensorView, 4, 3,
        {4, 4},
        [](auto&& buf, int i, int j) {
            buf(i, j) = 4 * i + j;
        },
        {4, 4, 3}
    );
    ASSERT_EQ(input.dimensions(), 2);
    ASSERT_EQ(func.dimensions(), 3);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                bool inRange = (i + k) >= 1 && (i + k) < 5;
                if (inRange)
                    ASSERT_EQ(outputBuffer(i, j, k), 4 * (i + k - 1) + j);
                else
                    ASSERT_EQ(outputBuffer(i, j, k), 0);
            }
        }
    }
}

TEST_F(transforms_tests, merge) {
    MergeOp mergeOpL { dimCH, Order::Left, sizeH }, mergeOpR { dimCH, Order::Right, sizeH };
    Interface in { &mergeOpL, &mergeOpR, dimH, dimW };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].accessToString(ctx), "[(i_2)/(H),(i_2)%(H),i_0,i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[c,H,H,W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[(i_2)/(H),(i_2)%(H),i_0,i_1];
        }
    }
}
)");
    auto [input, func, outputBuffer] = realize(
        tensorView, 4, 2,
        {2, 4, 4, 4},
        [](auto&& buf, int i, int j, int k, int l) {
            buf(i, j, k, l) = 4 * i + j;
        },
        {4, 4, 8}
    );
    ASSERT_EQ(input.dimensions(), 4);
    ASSERT_EQ(func.dimensions(), 3);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), k);
            }
        }
    }
}

TEST_F(transforms_tests, split) {
    SplitOp splitOp { dimH, dimW };
    Interface in { &splitOp, dimCH };
    auto tensorView = TensorView { { in } };
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getShape().toString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].accessToString(ctx), "[((i_0)*(W))+(i_1),i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0].shapeToString(ctx), "[H*W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = in_0[((i_0)*(W))+(i_1),i_2];
        }
    }
}
)");
    auto [input, func, outputBuffer] = realize(
        tensorView, 4, 2,
        {16, 8},
        [](auto&& buf, int i, int j) {
            buf(i, j) = i;
        },
        {4, 4, 8}
    );
    ASSERT_EQ(input.dimensions(), 2);
    ASSERT_EQ(func.dimensions(), 3);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                ASSERT_EQ(outputBuffer(i, j, k), 4 * i + j);
            }
        }
    }
}

TEST_F(transforms_tests, dimension_store) {
    DimensionStore store;
    Dimension s1 = store.get<ShiftOp>(dimH, 1);
    Dimension s2 = store.get<ShiftOp>(dimH, 1);
    ASSERT_EQ(s1, s2);
}

} // namespace kas
