#include <memory>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "KAS/Search/ShapeNode.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

TEST(search_tests, shape_node) {
    auto ctx = BindingContext { static_cast<std::size_t>(2), static_cast<std::size_t>(0) };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    auto shape = Shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW } };
    ShapeNode n1 { std::move(shape), false };
    ShapeNode::Next p1 { std::make_unique<ShareShapeOp>(0, 1, 0) };
    p1.node.reset(new ShapeNode { p1.shapeOp->transformShapeInverse(n1.shape), false });
    ShapeNode& n2 = *p1.node;
    ShapeNode::Next p2 { std::make_unique<MapReduceShapeOp>(0, *sizeH * *sizeW, Manipulation::MapType::Identity, Manipulation::ReduceType::Sum) };
    p2.node.reset(new ShapeNode { p2.shapeOp->transformShapeInverse(n2.shape), true });
    ShapeNode& n3 = *p2.node;
    auto cgCtx = std::make_shared<CodeGenContext>();
    auto tensorView = TensorView { n3.shape, cgCtx };
    p2.shapeOp->transformTensor(tensorView);
    p1.shapeOp->transformTensor(tensorView);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_1,i_2] with Identity mapped with i_0 Sum reduced");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[x_0,x_1] with reduced [x_0*x_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[i_0,i_1,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[x_0*x_1,x_0,x_0,x_1]");
}

TEST(search_tests, sample) {
    SampleOptions options;
    options.countPrimaryVariables = 4;
    options.countCoefficientVariables = 5;
    options.depth = 2;
    options.dimLowerBound = 4;
    options.dimUpperBound = 8;
    Sampler sampler("[H,W]", "[N,C,H,W]", options);
    auto& ctx = sampler.getBindingContext();
    auto callback = [&](TensorView tensorView) {
        std::cout << "Input Shape: ";
        bool first = true;
        for (const auto& tensor: tensorView.getUnderlyingTensors()) {
            if (first) {
                first = false;
            } else {
                std::cout << ", ";
            }
            std::cout << tensor->shapeToString(ctx);
        }
        std::cout << tensorView.printNestedLoops(ctx);
    };
    for (int i = 0; i < 10; ++i) {
        auto [sample, path] = sampler.randomSample();
        ASSERT_EQ(Shape::concat(sample.getInputShapes()).toString(ctx), sampler.visit(path).shape.toString(ctx));
        callback(sample);
    }
}

} // namespace kas
