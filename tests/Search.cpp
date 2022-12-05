#include <memory>
#include <gtest/gtest.h>
#include <vector>

#include "KAS/Search/ShapeNode.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/IteratorEvaluator.hpp"


using namespace kas;

TEST(search_tests, shape_node) {
    auto ctx = BindingContext { 2, 0 };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    auto shape = Shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW } };
    std::shared_ptr<ShapeNode> root = std::make_shared<ShapeNode>(shape);
    auto node = std::make_shared<ShapeNode>(root, std::move(std::make_unique<ShareShapeOp>(0, 1, 0)));
    auto tensorView = node->buildTensorView();
    auto evaluator = IteratorEvaluator { ctx };
    evaluator.evaluateTensorAccess(tensorView);
    ASSERT_EQ(tensorView.tensor->accessToString(), "[i_0,i_0,i_1]");
    ASSERT_EQ(tensorView.tensor->shapeToString(ctx), "[x_0,x_0,x_1]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[x_0,x_1]");
}
