#include <memory>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "KAS/Core/Representation.hpp"
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
    Representation repr { ctx };
    repr.addShape(n3.shape);
    repr.addTransform(p2.shapeOp->transformTensor(tensorView));
    repr.addShape(n2.shape);
    repr.addTransform(p1.shapeOp->transformTensor(tensorView));
    repr.addShape(n1.shape);
    tensorView.finishConstruction();
    tensorView.setDefaultInterfaceAccess();
    tensorView.evaluateTensorAccess();
    ASSERT_EQ(tensorView.actualAccessToString(ctx, *cgCtx), "[i_1,i_2] with Identity mapped with i_0 Sum reduced");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[x_0,x_1] with reduced [x_0*x_1]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->interfaceAccessToString(ctx, *cgCtx), "[i_0,i_1,i_1,i_2]");
    ASSERT_EQ(tensorView.getUnderlyingTensors()[0]->shapeToString(ctx), "[x_0*x_1,x_0,x_0,x_1]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx),
R"(for (int i_1 = 0; i_1 < x_0; i_1++) {
    for (int i_2 = 0; i_2 < x_1; i_2++) {
        float temp_i_0 = 0;
        for (int i_0 = 0; i_0 < x_0*x_1; i_0++) {
            temp_i_0 += Identity(t[i_0,i_1,i_1,i_2]);
        }
        out[i_1,i_2] = temp_i_0;
    }
}
)");
    ASSERT_EQ(repr.description(), "[x_0*x_1,x_0,x_0,x_1]\nMapReduce Identity Sum 0\n[x_0,x_0,x_1]\nShare 0, 1 -> 0\n[x_0,x_1]\n");
}

TEST(search_tests, sample) {
    SampleOptions options;
    options.seed = 42;
    options.depth = 2;
    options.dimLowerBound = 4;
    options.dimUpperBound = 8;
    Sampler sampler("[H,W]", "[N,C,H,W]", {}, {"k_1", "s_1", "k_2", "s_2"}, options);
    auto& ctx = sampler.getBindingContext();
    auto callback = [&](TensorView& tensorView, Representation& repr) {
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
        std::cout << std::endl;
        std::cout << tensorView.printNestedLoops(ctx);
        std::cout << repr.description();
    };
    for (int i = 0; i < 10; ++i) {
        auto path = sampler.randomPathWithPrefix({});
        auto [sample, cgCtx, repr] = sampler.realize(path);
        ASSERT_EQ(Shape::concat(sample.getInputShapes()).toString(ctx), sampler.visit(path).shape.toString(ctx));
        callback(sample, repr);
    }
}

} // namespace kas
