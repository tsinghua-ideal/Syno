#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Halide.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Transforms/Finalize.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

// Here, we verify the shape, iterator and codegen semantics of each transform.
class transforms_tests: public ::testing::Test {
protected:
    using SizeName = BindingContext::Metadata;
    BindingContext ctx { std::vector<SizeName>{ SizeName("H", 128), SizeName("W", 128) }, std::vector<SizeName>{ SizeName("c", 5) } };
    std::shared_ptr<Size> sizeH = ctx.getSinglePrimaryVariableSize(0);
    std::shared_ptr<Size> sizeW = ctx.getSinglePrimaryVariableSize(1);
    std::shared_ptr<Size> sizeC = ctx.getSingleCoefficientVariableSize(0);
    Shape shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW, *sizeC * *sizeH } };

    std::pair<TensorView, std::shared_ptr<CodeGenContext>> applyOp(const PrimitiveShapeOp& op, const Shape& shape) const {
        auto cgCtx = std::make_shared<CodeGenContext>();
        auto tensorView = TensorView { op.transformShapeInverse(shape), cgCtx };
        op.transformTensor(tensorView);
        tensorView.finishConstruction();
        tensorView.setDefaultInterfaceAccess();
        tensorView.evaluateTensorAccess();
        return { std::move(tensorView), std::move(cgCtx) };
    }

    std::pair<TensorView, std::shared_ptr<CodeGenContext>> applyOp(const PrimitiveShapeOp& op) const {
        return applyOp(op, shape);
    }

    template<std::size_t OutputDimensions>
    struct HalideEssentials {
        Halide::ImageParam input;
        Halide::Func func;
        Halide::Buffer<float, OutputDimensions> outputBuffer;
    };

    template<std::size_t InputDimensions, std::size_t OutputDimensions>
    HalideEssentials<OutputDimensions> realize(
        const TensorView& tensorView,
        std::size_t primaryDim, std::size_t coefficientDim,
        const int (&inputDimensions)[InputDimensions],
        auto&& inputInitializer,
        const int (&outputDimensions)[OutputDimensions]
    ) const {
        HalideGen gen(ctx, tensorView);
        auto [inputs, func] = gen.createFunc("semantic_test");
        KAS_ASSERT(inputs.size() == 1);
        auto& input = inputs[0];
        Halide::ParamMap params;
        for (auto& param: gen.primaryConsts) {
            params.set(param, static_cast<int>(primaryDim));
        }
        for (auto& param: gen.coefficientConsts) {
            params.set(param, static_cast<int>(coefficientDim));
        }
        auto inputBuffer = Halide::Buffer<float, InputDimensions>(std::vector<int>(inputDimensions, inputDimensions + InputDimensions));
        inputBuffer.for_each_element(std::bind_front(inputInitializer, std::ref(inputBuffer)));
        input.set(inputBuffer);
        func.compute_root();
        Halide::Buffer<float, OutputDimensions> outputBuffer = func.realize(std::vector<int>(outputDimensions, outputDimensions + OutputDimensions), Halide::get_host_target(), params);
        return { std::move(input), std::move(func), std::move(outputBuffer) };
    }
};

TEST_F(transforms_tests, share) {
    ShareShapeOp shareOp { 0, 1, 0 };
    auto [tensorView, cgCtx] = applyOp(shareOp);
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[i_0,i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H,H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = t[i_0,i_0,i_1,i_2];
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
    MapReduceShapeOp mapReduceOp { 0, *sizeH * *sizeW, Manipulation::MapType::ReLU, Manipulation::ReduceType::Sum };
    auto [tensorView, cgCtx] = applyOp(mapReduceOp);
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_1,i_2,i_3] with ReLU mapped with i_0 Sum reduced");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,c*H] with reduced [H*W]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2,i_3]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H*W,H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_1 = 0; i_1 < H; i_1++) {
    for (int i_2 = 0; i_2 < W; i_2++) {
        for (int i_3 = 0; i_3 < c*H; i_3++) {
            float temp_i_0 = 0;
            for (int i_0 = 0; i_0 < H*W; i_0++) {
                temp_i_0 += ReLU(t[i_0,i_1,i_2,i_3]);
            }
            out[i_1,i_2,i_3] = temp_i_0;
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
    ShiftShapeOp shiftOp { 0, 0, 1 };
    auto [tensorView, cgCtx] = applyOp(shiftOp);
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[(((i_0)+(1))+(H))%(H),i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = t[(((i_0)+(1))+(H))%(H),i_1,i_2];
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
    StrideShapeOp strideOp { 0, 0, sizeC };
    auto [tensorView, cgCtx] = applyOp(strideOp);
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[(c)*(i_0),i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[c*H,W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = t[(c)*(i_0),i_1,i_2];
        }
    }
}
)");
}

TEST_F(transforms_tests, unfold) {
    Shape shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW, sizeC } };
    UnfoldShapeOp unfoldOp { 0, 0, 2 };
    auto [tensorView, cgCtx] = applyOp(unfoldOp, shape);
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,c]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[((i_0)+(i_2))-(((c)-(1))/(2)),i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H,W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c; i_2++) {
            out[i_0,i_1,i_2] = t[((i_0)+(i_2))-(((c)-(1))/(2)),i_1];
        }
    }
}
)");
}

TEST_F(transforms_tests, merge) {
    MergeShapeOp mergeOp { 0, 1, 2, sizeH };
    auto [tensorView, cgCtx] = applyOp(mergeOp);
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[(i_2)/(H),(i_2)%(H),i_0,i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[c,H,H,W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = t[(i_2)/(H),(i_2)%(H),i_0,i_1];
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
    SplitShapeOp splitOp { 0, 0, 1 };
    auto [tensorView, cgCtx] = applyOp(splitOp);
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,c*H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[((i_0)*(W))+(i_1),i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H*W,c*H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < c*H; i_2++) {
            out[i_0,i_1,i_2] = t[((i_0)*(W))+(i_1),i_2];
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

TEST_F(transforms_tests, finalize) {
    Shape outputShape { std::vector<std::shared_ptr<Size>> { *(*sizeH * *sizeW) / *sizeC, *sizeC * *sizeW } };
    Shape desired { std::vector<std::shared_ptr<Size>> { sizeH, sizeW } };
    FinalizeShapeOp finalizeOp(outputShape, desired, FinalizeShapeOp::Epilogue {
        { 0, 0 }, // Map the inputs H and W to group 0
        { { 0, 1 } }, // Group 0 consists of c^-1*H*W and c*W
    });
    auto [tensorView, cgCtx] = applyOp(finalizeOp, outputShape);
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[c^-1*H*W,c*W]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H,W,W]");
    // Note that the remainder of a size is placed frontmost. So when merging we are actually doing [W,H,W], where the last dimension, which is the remainder, is promoted to the first of the group.
    // First a split is evaluated, [((i_0)*(c*W))+(i_1)]
    // Then a merge ((the fusion of remainder W and input H) and input W),
    // [(((i_0)*(c*W))+(i_1))/(W),(((i_0)*(c*W))+(i_1))%(W)]
    // Then yet another merge (remainder W and input H),
    // [((((i_0)*(c*W))+(i_1))/(W))/(H),((((i_0)*(c*W))+(i_1))/(W))%(H),(((i_0)*(c*W))+(i_1))%(W)]
    // In the original order in the input tensor,
    // [((((i_0)*(c*W))+(i_1))/(W))%(H),(((i_0)*(c*W))+(i_1))%(W),((((i_0)*(c*W))+(i_1))/(W))/(H)]
    ASSERT_EQ(
        tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx),
        "[((((i_0)*(c*W))+(i_1))/(W))%(H),(((i_0)*(c*W))+(i_1))%(W),((((i_0)*(c*W))+(i_1))/(W))/(H)]"
    );
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < c^-1*H*W; i_0++) {
    for (int i_1 = 0; i_1 < c*W; i_1++) {
        out[i_0,i_1] = t[((((i_0)*(c*W))+(i_1))/(W))%(H),(((i_0)*(c*W))+(i_1))%(W),((((i_0)*(c*W))+(i_1))/(W))/(H)];
    }
}
)");
    auto [input, func, outputBuffer] = realize(
        tensorView, 4, 2,
        {4, 4, 4},
        [](auto&& buf, int i, int j, int k) {
            buf(i, j, k) = k * 16 + i * 4 + j;
        },
        {8, 8}
    );
    ASSERT_EQ(input.dimensions(), 3);
    ASSERT_EQ(func.dimensions(), 2);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            ASSERT_EQ(outputBuffer(i, j), 8 * i + j);
        }
    }
}

TEST_F(transforms_tests, finalize_gen) {
    BindingContext ctx { std::vector<SizeName>{ SizeName("N", 128), SizeName("C", 3), SizeName("H", 128), SizeName("W", 128) }, std::vector<SizeName>{ SizeName("k", 5), SizeName("s", 2) } };
    std::shared_ptr<Size> primaryN = ctx.getSinglePrimaryVariableSize(0);
    std::shared_ptr<Size> primaryC = ctx.getSinglePrimaryVariableSize(1);
    std::shared_ptr<Size> primaryH = ctx.getSinglePrimaryVariableSize(2);
    std::shared_ptr<Size> primaryW = ctx.getSinglePrimaryVariableSize(3);
    std::shared_ptr<Size> coeffK = ctx.getSingleCoefficientVariableSize(0);
    std::shared_ptr<Size> coeffS = ctx.getSingleCoefficientVariableSize(1);
    Shape outputShape { std::vector<std::shared_ptr<Size>> {
        primaryN, *primaryW / *coeffK, primaryH, *primaryC / *coeffK, primaryW, coeffK, coeffS
    }};
    Shape desiredShape { std::vector<std::shared_ptr<Size>> {
        primaryW, coeffS, *primaryW * *coeffK
    }};
    std::vector<std::size_t> mappings {
        1, 3, 4
    };
    auto epilogue = FinalizeShapeOp::solveWithMappings(outputShape, desiredShape, mappings).value();
    std::cout << epilogue.toDebugString(ctx, outputShape, desiredShape);
    std::cout << "\nNow automatically generate epilogue:\n" << std::endl;
    for (auto&& e: FinalizeShapeOp::generate(outputShape, { .desired = desiredShape })) {
        std::cout << e->getEpilogue().toDebugString(ctx, outputShape, desiredShape);
        auto tensorView = TensorView { e->transformShapeInverse(outputShape), std::make_shared<CodeGenContext>() };
        e->transformTensor(tensorView);
        tensorView.finishConstruction();
        tensorView.setDefaultInterfaceAccess();
        tensorView.evaluateTensorAccess();
    }
}

} // namespace kas
