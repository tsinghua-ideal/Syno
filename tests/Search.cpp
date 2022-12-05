#include <memory>
#include <gtest/gtest.h>
#include <vector>

#include "KAS/Search/ShapeNode.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Transforms/Reduce.hpp"
#include "KAS/Core/Tensor.hpp"


using namespace kas;

TEST(search_tests, shape_node) {
    auto ctx = BindingContext { 2, 0 };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    auto shape = Shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW } };
    std::shared_ptr<ShapeNode> root = std::make_shared<ShapeNode>(shape);
    auto node1 = std::make_shared<ShapeNode>(root, std::move(std::make_unique<ShareShapeOp>(0, 1, 0)));
    auto node2 = std::make_shared<ShapeNode>(node1, std::move(std::make_unique<ReduceShapeOp>(0, *sizeH * *sizeW)));
    auto tensorView = node2->buildTensorView();
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.accessToString(), "[i_0,i_1] with reduced [i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[x_0,x_1] with reduced [x_0x_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->accessToString(), "[i_2,i_0,i_0,i_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[x_0x_1,x_0,x_0,x_1]");
}
