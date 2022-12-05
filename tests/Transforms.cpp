#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms.hpp"
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
    auto tensorView = TensorView { shareOp.transformShapeInverse(shape) };
    shareOp.transformTensor(tensorView);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.accessToString(), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->accessToString(), "[i_0,i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[H,H,W,(c)H]");
}

TEST_F(transforms_tests, reduce) {
    ReduceShapeOp reduceOp { 0, *sizeH * *sizeW, ReduceManipulation::Type::Sum };
    auto tensorView = TensorView { reduceOp.transformShapeInverse(shape) };
    reduceOp.transformTensor(tensorView);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.accessToString(), "[i_0,i_1,i_2] with reduced [i_3]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H] with reduced [HW]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->accessToString(), "[i_3,i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[HW,H,W,(c)H]");
}

TEST_F(transforms_tests, map) {
    MapShapeOp mapOp { MapManipulation::Type::ReLU };
    auto tensorView = TensorView { mapOp.transformShapeInverse(shape) };
    mapOp.transformTensor(tensorView);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.accessToString(), "[i_0,i_1,i_2] with mapped [ReLU]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->accessToString(), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[H,W,(c)H]");
}

TEST_F(transforms_tests, shift) {
    ShiftShapeOp shiftOp { 0, 0, 1 };
    auto tensorView = TensorView { shiftOp.transformShapeInverse(shape) };
    shiftOp.transformTensor(tensorView);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.accessToString(), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->accessToString(), "[(i_0+1+H)%(H),i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[H,W,(c)H]");
}

TEST_F(transforms_tests, stride) {
    StrideShapeOp strideOp { 0, 0, sizeC };
    auto tensorView = TensorView { strideOp.transformShapeInverse(shape) };
    strideOp.transformTensor(tensorView);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.accessToString(), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)H]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->accessToString(), "[(c)1*i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[(c)H,W,(c)H]");
}

TEST_F(transforms_tests, unfold) {
    Shape shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW, sizeC } };
    UnfoldShapeOp unfoldOp { 0, 0, 2 };
    auto tensorView = TensorView { unfoldOp.transformShapeInverse(shape) };
    unfoldOp.transformTensor(tensorView);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.accessToString(), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[H,W,(c)1]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->accessToString(), "[(i_0+i_2-(c)1/2),i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[H,W]");
}
