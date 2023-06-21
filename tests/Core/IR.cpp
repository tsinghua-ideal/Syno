#include "gtest/gtest.h"

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Lower.hpp"


using namespace kas;

TEST(core_ir_tests, tensor_expression) {
    auto in0 = TensorTensorExpression::Create(TensorExpression::Input<0>);
    ASSERT_EQ(in0.toString(), "in_0");
    auto in1 = TensorTensorExpression::Create(TensorExpression::Input<1>);
    ASSERT_EQ(in1.toString(), "in_1");
    auto p = in0 * in1;
    ASSERT_EQ(p.toString(), "in_0 * in_1");
    auto d0 = TensorExpressionDifferentiator { 0 };
    ASSERT_EQ(d0.differentiate(in0).toString(), "1");
    auto d1 = TensorExpressionDifferentiator { 1 };
    ASSERT_EQ(d1.differentiate(in1).toString(), "1");
    auto din0 = d0.differentiate(p);
    ASSERT_EQ(din0.toString(), "in_1");
    auto din1 = d1.differentiate(p);
    ASSERT_EQ(din1.toString(), "in_0");

    auto in2 = TensorTensorExpression::Create(TensorExpression::Input<2>);
    auto p3 = in0 * in1 * in2;
    ASSERT_EQ(p3.toString(), "in_0 * in_1 * in_2");
    auto d2 = TensorExpressionDifferentiator { 2 };
    ASSERT_EQ(d2.differentiate(in2).toString(), "1");
    auto din2 = d2.differentiate(p3);
    ASSERT_EQ(din2.toString(), "in_0 * in_1");
    ASSERT_EQ(d0.differentiate(p3).toString(), "in_1 * in_2");
    ASSERT_EQ(d1.differentiate(p3).toString(), "in_0 * in_2");
}
