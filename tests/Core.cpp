#include <memory>
#include <vector>
#include <gtest/gtest.h>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/IteratorEvaluator.hpp"


using namespace kas;

TEST(core_tests, size) {
    std::vector<BindingContext::Metadata> metaPrimary { BindingContext::Metadata("H"), BindingContext::Metadata("W") };
    auto ctx = BindingContext { std::move(metaPrimary), std::vector<BindingContext::Metadata>() };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    ASSERT_EQ(sizeH->toString(ctx), "H");
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    ASSERT_EQ(sizeW->toString(ctx), "W");
    auto sizeHW = Size { std::vector<int> { 1, 1 }, std::vector<int>() };
    ASSERT_EQ(*sizeH * *sizeW, sizeHW);
    ASSERT_EQ(sizeHW.toString(ctx), "HW");
}

TEST(core_tests, tensor) {
    auto ctx = BindingContext { 3, 2 };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    ASSERT_EQ(sizeH->toString(ctx), "x_0");
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    ASSERT_EQ(sizeW->toString(ctx), "x_1");
    auto sizeC = ctx.getSingleCoefficientVariableSize(0);
    ASSERT_EQ(sizeC->toString(ctx), "(c_0)1");
    auto shape = Shape { std::vector<std::shared_ptr<Size>> { sizeH, sizeW, sizeC } };
    auto tensor = std::make_shared<PureTensor>(shape);
    auto tensorView = TensorView { tensor };
    auto evaluator = IteratorEvaluator { ctx };
    evaluator.evaluateTensorAccess(tensorView);
    ASSERT_EQ(tensor->accessToString(), "[i_0,i_1,i_2]");
}
