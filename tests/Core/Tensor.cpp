#include <gtest/gtest.h>

#include "KAS/Core/Lower.hpp"
#include "KAS/Core/Tensor.hpp"


using namespace kas;

TEST(core_tensor_tests, tensor) {
    auto ctx = BindingContext(3, 2);
    auto sizeX0 = ctx.getSinglePrimaryVariableSize(0);
    ASSERT_EQ(sizeX0.toString(ctx), "x_0");
    auto sizeX1 = ctx.getSinglePrimaryVariableSize(1);
    ASSERT_EQ(sizeX1.toString(ctx), "x_1");
    auto sizeC0 = ctx.getSingleCoefficientVariableSize(0);
    ASSERT_EQ(sizeC0.toString(ctx), "c_0");
    auto sizeC1 = ctx.getSingleCoefficientVariableSize(1);
    ASSERT_EQ(sizeC1.toString(ctx), "c_1");
    Iterator i_0 { 0, sizeX0 }, i_1 { 1, sizeX1 }, i_2 { 2, sizeC0 };
    MapReduce i_3 { 0, sizeC1, MapReduce::MapType::Identity, MapReduce::ReduceType::Sum };
    Interface interface { &i_0, &i_1, &i_2, &i_3 };
    auto tensorView = TensorView({ interface }, TensorExpression::ProductOfTensors(1));
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
