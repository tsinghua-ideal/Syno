#include <map>
#include <memory>
#include <ranges>
#include <string>
#include <vector>

#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/ShapeNode.hpp"
#include "KAS/Transforms.hpp"


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
}

TEST(search_tests, sample) {
    SampleOptions options;
    options.seed = 42;
    options.depth = 4;
    options.dimLowerBound = 4;
    options.dimUpperBound = 8;
    Sampler sampler("[H,W]", "[N,C,H,W]", {}, {"k_1", "s_1", "k_2", "s_2"}, options);
    auto& ctx = sampler.getBindingContext();
    ASSERT_EQ(ctx.getPrimaryCount(), 4);
    ASSERT_EQ(ctx.getCoefficientCount(), 4);
    for (int i = 0; i < 10; ++i) {
        auto path = sampler.randomPathWithPrefix({});
        auto [tensorView, cgCtx] = sampler.realize(path);
        ASSERT_EQ(tensorView.getFusedInputShapes().toString(ctx), sampler.visit(path).shape.toString(ctx));

        auto r = tensorView.getUnderlyingTensors() | std::ranges::views::transform([&](const auto& tensor) { return tensor->shapeToString(ctx); });
        std::cout << fmt::format("Input Shape: {}", fmt::join(r, ", ")) << std::endl;
        std::cout << tensorView.printNestedLoops(ctx);

        constexpr int dimH = 4, dimW = 4, dimN = 4, dimC = 4, dimK1 = 3, dimS1 = 2, dimK2 = 3, dimS2 = 2;
        HalideGen gen(ctx, tensorView);
        auto name = "search_codegen_test_" + std::to_string(i);
        auto [inputs, func] = gen.createFunc(name);
        ASSERT_EQ(inputs.size(), 2);
        auto& input = inputs[0], & weight = inputs[1];
        std::map<std::string, std::size_t> dict { { "H", dimH }, { "W", dimW }, { "N", dimN }, { "C", dimC }, { "k_1", dimK1 }, { "s_1", dimS1 }, { "k_2", dimK2 }, { "s_2", dimS2 } };
        auto getInputsShape = [&](int i) -> std::vector<std::size_t> {
            return ctx.evaluateShape(tensorView.getUnderlyingTensors()[i]->getShapeRef(), dict);
        };
        auto inputShape = getInputsShape(0), weightShape = getInputsShape(1);
        auto expectedInputShape = std::vector<std::size_t> {dimH, dimW};
        ASSERT_EQ(inputShape, expectedInputShape);
        Halide::ParamMap params = gen.getParamMap(dict);
        // Revert the order of shape due to column-major layout.
        auto inputBuffer = Halide::Buffer<float, 2>(std::vector<int>(inputShape.rbegin(), inputShape.rend()));
        inputBuffer.for_each_value([=](float& v) {
            static int cnt = 0;
            v = cnt++;
        });
        input.set(inputBuffer);
        auto weightBuffer = Halide::Buffer<float>(std::vector<int>(weightShape.rbegin(), weightShape.rend()));
        weightBuffer.for_each_value([=](float& v) {
            static int cnt = 0;
            v = cnt++;
        });
        weight.set(weightBuffer);
        func.compute_root();
        func.trace_stores();
        auto outputShape = std::vector<int> {dimN, dimC, dimH, dimW};
        Halide::Buffer<float, 4> outputBuffer = func.realize(std::vector<int>(outputShape.rbegin(), outputShape.rend()), Halide::get_host_target(), params);
    }
}

} // namespace kas
