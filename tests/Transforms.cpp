#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Transforms/Finalize.hpp"
#include "KAS/Transforms/Share.hpp"


using namespace kas;

class transforms_tests: public ::testing::Test {
protected:
    using SizeName = BindingContext::Metadata;
    BindingContext ctx { std::vector<SizeName>{ SizeName("H", 128), SizeName("W", 128) }, std::vector<SizeName>{ SizeName("c", 5) } };
    std::shared_ptr<CodeGenContext> cgCtx = std::make_shared<CodeGenContext>();
    std::shared_ptr<Size> sizeH = ctx.getSinglePrimaryVariableSize(0);
    std::shared_ptr<Size> sizeW = ctx.getSinglePrimaryVariableSize(1);
    std::shared_ptr<Size> sizeC = ctx.getSingleCoefficientVariableSize(0);
    Shape shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW, *sizeC * *sizeH } };
};

TEST_F(transforms_tests, share) {
    ShareShapeOp shareOp { 0, 1, 0 };
    auto tensorView = TensorView { shareOp.transformShapeInverse(shape), cgCtx };
    shareOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[i_0,i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H,H,W,(c)H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < (c)H; i_2++) {
            out[i_0,i_1,i_2] = t[i_0,i_0,i_1,i_2];
        }
    }
}
)");
}

TEST_F(transforms_tests, map_reduce) {
    MapReduceShapeOp mapReduceOp { 0, *sizeH * *sizeW, Manipulation::MapType::ReLU, Manipulation::ReduceType::Sum };
    auto tensorView = TensorView { mapReduceOp.transformShapeInverse(shape), cgCtx };
    mapReduceOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_1,i_2,i_3] with ReLU mapped with i_0 Sum reduced");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H] with reduced [HW]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2,i_3]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[HW,H,W,(c)H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_1 = 0; i_1 < H; i_1++) {
    for (int i_2 = 0; i_2 < W; i_2++) {
        for (int i_3 = 0; i_3 < (c)H; i_3++) {
            float temp_i_0 = 0;
            for (int i_0 = 0; i_0 < HW; i_0++) {
                temp_i_0 += ReLU(t[i_0,i_1,i_2,i_3]);
            }
            out[i_1,i_2,i_3] = temp_i_0;
        }
    }
}
)");
}

TEST_F(transforms_tests, shift) {
    ShiftShapeOp shiftOp { 0, 0, 1 };
    auto tensorView = TensorView { shiftOp.transformShapeInverse(shape), cgCtx };
    shiftOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[(((i_0)+(1))+(H))%(H),i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < (c)H; i_2++) {
            out[i_0,i_1,i_2] = t[(((i_0)+(1))+(H))%(H),i_1,i_2];
        }
    }
}
)");
}

TEST_F(transforms_tests, stride) {
    StrideShapeOp strideOp { 0, 0, sizeC };
    auto tensorView = TensorView { strideOp.transformShapeInverse(shape), cgCtx };
    strideOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[((c)1)*(i_0),i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[(c)H,W,(c)H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < (c)H; i_2++) {
            out[i_0,i_1,i_2] = t[((c)1)*(i_0),i_1,i_2];
        }
    }
}
)");
}

TEST_F(transforms_tests, unfold) {
    Shape shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW, sizeC } };
    UnfoldShapeOp unfoldOp { 0, 0, 2 };
    auto tensorView = TensorView { unfoldOp.transformShapeInverse(shape), cgCtx };
    unfoldOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)1]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[((i_0)+(i_2))-(((c)1)/(2)),i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H,W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < (c)1; i_2++) {
            out[i_0,i_1,i_2] = t[((i_0)+(i_2))-(((c)1)/(2)),i_1];
        }
    }
}
)");
}

TEST_F(transforms_tests, merge) {
    MergeShapeOp mergeOp { 0, 1, 2, sizeH };
    auto tensorView = TensorView { mergeOp.transformShapeInverse(shape), cgCtx };
    mergeOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[(i_2)/(H),(i_2)%(H),i_0,i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[(c)1,H,H,W]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < (c)H; i_2++) {
            out[i_0,i_1,i_2] = t[(i_2)/(H),(i_2)%(H),i_0,i_1];
        }
    }
}
)");
}

TEST_F(transforms_tests, split) {
    SplitShapeOp splitOp { 0, 0, 1 };
    auto tensorView = TensorView { splitOp.transformShapeInverse(shape), cgCtx };
    splitOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[((i_0)*(W))+(i_1),i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[HW,(c)H]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < H; i_0++) {
    for (int i_1 = 0; i_1 < W; i_1++) {
        for (int i_2 = 0; i_2 < (c)H; i_2++) {
            out[i_0,i_1,i_2] = t[((i_0)*(W))+(i_1),i_2];
        }
    }
}
)");
}

TEST_F(transforms_tests, finalize) {
    Shape outputShape { std::vector<std::shared_ptr<Size>> { *(*sizeH * *sizeW) / *sizeC, *sizeC * *sizeW } };
    Shape desired { std::vector<std::shared_ptr<Size>> { sizeH, sizeW } };
    FinalizeShapeOp finalizeOp(outputShape, desired, FinalizeShapeOp::Epilogue { { 0, 0 }, { { 0, 1 } } });
    auto tensorView = TensorView { finalizeOp.transformShapeInverse(outputShape), cgCtx };
    finalizeOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_0,i_1]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[(1/c)HW,(c)W]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[H,W,W]");
    // Note that the remainder of a size is placed frontmost. So when merging we are actually doing [W,H,W], where the last dimension, which is the remainder, is promoted to the first of the group.
    // First a split is evaluated, [((i_0)*((c)W))+(i_1)]
    // Then a merge ((the fusion of remainder W and input H) and input W),
    // [(((i_0)*((c)W))+(i_1))/(W),(((i_0)*((c)W))+(i_1))%(W)]
    // Then yet another merge (remainder W and input H),
    // [((((i_0)*((c)W))+(i_1))/(W))/(H),((((i_0)*((c)W))+(i_1))/(W))%(H),(((i_0)*((c)W))+(i_1))%(W)]
    // In the original order in the input tensor,
    // [((((i_0)*((c)W))+(i_1))/(W))%(H),(((i_0)*((c)W))+(i_1))%(W),((((i_0)*((c)W))+(i_1))/(W))/(H)]
    ASSERT_EQ(
        tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx),
        "[((((i_0)*((c)W))+(i_1))/(W))%(H),(((i_0)*((c)W))+(i_1))%(W),((((i_0)*((c)W))+(i_1))/(W))/(H)]"
    );
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_0 = 0; i_0 < (1/c)HW; i_0++) {
    for (int i_1 = 0; i_1 < (c)W; i_1++) {
        out[i_0,i_1] = t[((((i_0)*((c)W))+(i_1))/(W))%(H),(((i_0)*((c)W))+(i_1))%(W),((((i_0)*((c)W))+(i_1))/(W))/(H)];
    }
}
)");
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
        auto tensorView = TensorView { e->transformShapeInverse(outputShape), cgCtx };
        e->transformTensor(tensorView);
        tensorView.finishConstruction();
        tensorView.setDefaultInterfaceAccess();
        tensorView.evaluateTensorAccess();
    }
}
