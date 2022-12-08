#include <memory>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "KAS/Search/ShapeNode.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Sample.hpp"


using namespace kas;

TEST(search_tests, shape_node) {
    auto ctx = BindingContext { 2, 0 };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    auto shape = Shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW } };
    std::shared_ptr<ShapeNode> root = std::make_shared<ShapeNode>(shape);
    auto node1 = std::make_shared<ShapeNode>(root, std::move(std::make_unique<ShareShapeOp>(0, 1, 0)));
    auto node2 = std::make_shared<ShapeNode>(node1, std::move(std::make_unique<ReduceShapeOp>(0, *sizeH * *sizeW, ReduceManipulation::Type::Sum)));
    auto remap = std::vector<int> { 3, 2, 1, 0 };
    auto tensorView = node2->buildTensorView(remap);
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.accessToString(), "[i_0,i_1] with reduced [i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[x_0,x_1] with reduced [x_0x_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->accessToString(), "[i_1,i_0,i_0,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensor()->shapeToString(ctx), "[x_1,x_0,x_0,x_0x_1]");
}

TEST(search_tests, sample) {
    SampleOptions options(4, 5, 4, 4, 8);
    Sampler sampler("[H,W]", "[N,C,H,W]", options);
    auto ctx = sampler.getBindingContext();
    auto callback = [&](TensorView tensorView) {
        std::cout << "Input Shape: " << tensorView.getUnderlyingTensor()->shapeToString(ctx) << std::endl;
        std::cout << "for (int i_0 = 0; i_0 < N; ++i_0)\n  for (int i_1 = 0; i_1 < C; ++i_1)\n    for (int i_2 = 0; i_2 < H; ++i_2)\n      for (int i_3 = 0; i_3 < W; ++i_3)\n        out[i_0,i_1,i_2,i_3]=t" << tensorView.getUnderlyingTensor()->accessToString() << std::endl;
    };
    sampler.sample(std::function(callback));
}
