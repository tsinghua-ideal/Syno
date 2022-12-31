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
    BindingContext ctx { std::vector<SizeName>{ SizeName("H"), SizeName("W") }, std::vector<SizeName>{ SizeName("c") } };
    std::shared_ptr<Size> sizeH = ctx.getSinglePrimaryVariableSize(0);
    std::shared_ptr<Size> sizeW = ctx.getSinglePrimaryVariableSize(1);
    std::shared_ptr<Size> sizeC = ctx.getSingleCoefficientVariableSize(0);
    Shape shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW, *sizeC * *sizeH } };
};

TEST_F(transforms_tests, share) {
    ShareShapeOp shareOp { 0, 1, 0 };
    auto tensorView = TensorView { shareOp.transformShapeInverse(shape), ctx };
    shareOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultAccesses(ctx);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->interfaceAccessToString(ctx), "[i_0,i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[H,H,W,(c)H]");
}

TEST_F(transforms_tests, reduce) {
    ReduceShapeOp reduceOp { 0, *sizeH * *sizeW, ReduceManipulation::Type::Sum };
    auto tensorView = TensorView { reduceOp.transformShapeInverse(shape), ctx };
    reduceOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultAccesses(ctx);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2] with reduced [ri_0]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H] with reduced [HW]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->interfaceAccessToString(ctx), "[ri_0,i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[HW,H,W,(c)H]");
}

TEST_F(transforms_tests, map) {
    MapShapeOp mapOp { MapManipulation::Type::ReLU };
    auto tensorView = TensorView { mapOp.transformShapeInverse(shape), ctx };
    mapOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultAccesses(ctx);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2] with mapped [ReLU]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->interfaceAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[H,W,(c)H]");
}

TEST_F(transforms_tests, shift) {
    ShiftShapeOp shiftOp { 0, 0, 1 };
    auto tensorView = TensorView { shiftOp.transformShapeInverse(shape), ctx };
    shiftOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultAccesses(ctx);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->interfaceAccessToString(ctx), "[(((i_0)+(1))+(H))%(H),i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[H,W,(c)H]");
}

TEST_F(transforms_tests, stride) {
    StrideShapeOp strideOp { 0, 0, sizeC };
    auto tensorView = TensorView { strideOp.transformShapeInverse(shape), ctx };
    strideOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultAccesses(ctx);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->interfaceAccessToString(ctx), "[((c)1)*(i_0),i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[(c)H,W,(c)H]");
}

TEST_F(transforms_tests, unfold) {
    Shape shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW, sizeC } };
    UnfoldShapeOp unfoldOp { 0, 0, 2 };
    auto tensorView = TensorView { unfoldOp.transformShapeInverse(shape), ctx };
    unfoldOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultAccesses(ctx);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)1]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->interfaceAccessToString(ctx), "[((i_0)+(i_2))-(((c)1)/(2)),i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[H,W]");
}

TEST_F(transforms_tests, merge) {
    MergeShapeOp mergeOp { 0, 1, 2, sizeH };
    auto tensorView = TensorView { mergeOp.transformShapeInverse(shape), ctx };
    mergeOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultAccesses(ctx);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->interfaceAccessToString(ctx), "[(i_2)/(H),(i_2)%(H),i_0,i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[(c)1,H,H,W]");
}

TEST_F(transforms_tests, split) {
    SplitShapeOp splitOp { 0, 0, 1 };
    auto tensorView = TensorView { splitOp.transformShapeInverse(shape), ctx };
    splitOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultAccesses(ctx);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->interfaceAccessToString(ctx), "[((i_0)*(W))+(i_1),i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[HW,(c)H]");
}

TEST_F(transforms_tests, finalize) {
    Shape outputShape { std::vector<std::shared_ptr<Size>> { *(*sizeH * *sizeW) / *sizeC, *sizeC * *sizeW } };
    Shape desired { std::vector<std::shared_ptr<Size>> { sizeH, sizeW } };
    FinalizeShapeOp finalizeOp(desired, FinalizeShapeOp::Epilogue { { 0, 0 }, { { 0, 1 } } });
    auto tensorView = TensorView { finalizeOp.transformShapeInverse(outputShape), ctx };
    finalizeOp.transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultAccesses(ctx);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[(1/c)HW,(c)W]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[H,W,W]");
    // Note that the remainder of a size is placed frontmost. So when merging we are actually doing [W,H,W], where the last dimension, which is the remainder, is promoted to the first of the group.
    // First a split is evaluated, [((i_0)*((c)W))+(i_1)]
    // Then a merge ((the fusion of remainder W and input H) and input W),
    // [(((i_0)*((c)W))+(i_1))/(W),(((i_0)*((c)W))+(i_1))%(W)]
    // Then yet another merge (remainder W and input H),
    // [((((i_0)*((c)W))+(i_1))/(W))/(H),((((i_0)*((c)W))+(i_1))/(W))%(H),(((i_0)*((c)W))+(i_1))%(W)]
    // In the original order in the input tensor,
    // [((((i_0)*((c)W))+(i_1))/(W))%(H),(((i_0)*((c)W))+(i_1))%(W),((((i_0)*((c)W))+(i_1))/(W))/(H)]
    ASSERT_EQ(
        tensorView.getUnderlyingTensor()->interfaceAccessToString(ctx),
        "[((((i_0)*((c)W))+(i_1))/(W))%(H),(((i_0)*((c)W))+(i_1))%(W),((((i_0)*((c)W))+(i_1))/(W))/(H)]"
    );
}

TEST_F(transforms_tests, finalize_gen) {
    BindingContext ctx { std::vector<SizeName>{ SizeName("N"), SizeName("C"), SizeName("H"), SizeName("W") }, std::vector<SizeName>{ SizeName("k"), SizeName("s") } };
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
    }
}
