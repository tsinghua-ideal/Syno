#include <gtest/gtest.h>

#include "KAS/Core/Lower.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Transforms/Transforms.hpp"


using namespace kas;

TEST(core_tensor_tests, tensor) {
    auto ctx = BindingContext(3, 2);
    ctx.debug();
    auto sizeX0 = ctx.getSinglePrimaryVariableSize(0);
    ASSERT_EQ(sizeX0.toString(ctx), "x_0");
    auto sizeX1 = ctx.getSinglePrimaryVariableSize(1);
    ASSERT_EQ(sizeX1.toString(ctx), "x_1");
    auto sizeC0 = ctx.getSingleCoefficientVariableSize(0);
    ASSERT_EQ(sizeC0.toString(ctx), "c_0");
    auto sizeC1 = ctx.getSingleCoefficientVariableSize(1);
    ASSERT_EQ(sizeC1.toString(ctx), "c_1");
    Iterator i_0 { 0, sizeX0 }, i_1 { 1, sizeX1 }, i_2 { 2, sizeC0 };
    ReduceOp i_3 { sizeC1, Reduce::ReduceType::Sum };
    auto tensorView = TensorView({{{&i_0, &i_1, &i_2, i_3.getInput(0)}, {}}}, TensorExpression::ProductOfTensors(1), ctx);
    ASSERT_EQ(tensorView.getInterfaceShape().toString(ctx), "[x_0, x_1, c_0]");
    ASSERT_EQ(tensorView.getForwardAccess().outerLoopsIteratorsToString(), "[i_0, i_1, i_2]");
    ASSERT_EQ(tensorView.getForwardAccess().innerLoopsIteratorsToString(), "[ri_0]");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, -1),
R"(for (int i_0 = 0; i_0 < x_0; i_0++) {
    for (int i_1 = 0; i_1 < x_1; i_1++) {
        for (int i_2 = 0; i_2 < c_0; i_2++) {
            float temp_ri_0 = 0;
            for (int ri_0 = 0; ri_0 < c_1; ri_0++) {
                temp_ri_0 += in_0[i_0, i_1, i_2, ri_0];
            }
            out[i_0, i_1, i_2] = temp_ri_0;
        }
    }
}
)");
    const auto& tensors = tensorView.getUnderlyingTensors();
    ASSERT_EQ(tensors.size(), 1);
    const auto& tensor = tensors[0];
    ASSERT_EQ(tensor.shapeToString(ctx), "[x_0, x_1, c_0, c_1]");
}

TEST(core_tensor_tests, subgraph) {
    auto ctx = BindingContext(3, 2);
    ctx.debug();
    auto [x_0, x_1, x_2, c_0, c_1] = ctx.getSizes("x_0", "x_1", "x_2", "c_0", "c_1");
    Iterator i_0 { 0, x_0 }, i_1 { 1, x_1 }, i_2 { 2, x_2 };
    GraphHandle interface = {{&i_0, &i_1, &i_2}, {}};

    auto [expansions, inputTensors, outputTensor] = IR::Build({{{&i_0, &i_1, &i_2}, {}}}, ctx);
    ASSERT_TRUE(std::ranges::all_of(expansions, [](const auto& expansion) { return expansion.empty(); }));
    ASSERT_EQ(inputTensors.size(), 1);
    ASSERT_EQ(inputTensors[0], outputTensor);
}

// This test cannot be done because only PyTorchGen requires this pattern. See `tests/IR/InterdependentShare.cpp`.
// TEST(core_tensor_tests, subgraph_diagonal) {
//     auto ctx = BindingContext(2, 0);
//     ctx.debug();
//     auto [x_0, x_1] = ctx.getSizes("x_0", "x_1");
//     Iterator i_0 { 0, x_0 };
//     ReduceOp i_1 { x_1, Reduce::ReduceType::Sum };
//     ShareOp shareOp { i_1.getInput(0) };
//     GraphHandle interface = {{&i_0, shareOp.getInputL(), shareOp.getInputR()}, {}};

//     auto [expansions, inputTensors, outputTensor] = IR::Build({{{&i_0, shareOp.getInputL(), shareOp.getInputR()}, {}}}, ctx);
//     ASSERT_TRUE(std::ranges::all_of(expansions, [](const auto& expansion) { return expansion.empty(); }));
//     ASSERT_EQ(inputTensors.size(), 1);
//     ASSERT_EQ(outputTensor.toString(ctx), "([x_0, x_1, x_1] -> [x_0])");
// }
