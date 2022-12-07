#include <memory>
#include <optional>
#include <vector>
#include <gtest/gtest.h>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Tensor.hpp"


using namespace kas;

TEST(core_tests, size) {
    std::vector<BindingContext::Metadata> metaPrimary { BindingContext::Metadata("H"), BindingContext::Metadata("W") };
    std::vector<BindingContext::Metadata> metaCoefficient { BindingContext::Metadata("c") };
    auto ctx = BindingContext { std::move(metaPrimary), std::move(metaCoefficient) };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    ASSERT_EQ(sizeH->toString(ctx), "H");
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    ASSERT_EQ(sizeW->toString(ctx), "W");
    auto sizeC = ctx.getSingleCoefficientVariableSize(0);
    auto sizeHWc = std::make_shared<Size>(std::vector<int> { 1, 1 }, std::vector<int> { -1 });
    ASSERT_EQ(*(*(*sizeH * *sizeW) / *sizeC), *sizeHWc);
    ASSERT_EQ(sizeHWc->toString(ctx), "(1/c)HW");
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
    auto tensorView = TensorView { tensor, std::nullopt };
    tensorView.evaluateTensorAccess(ctx);
    ASSERT_EQ(tensorView.accessToString(), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[x_0,x_1,(c_0)1]");
    ASSERT_EQ(tensor->accessToString(), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensor->shapeToString(ctx), "[x_0,x_1,(c_0)1]");
}

TEST(core_tests, parse_shape_names) {
    auto parsedNames1 = Shape::parseNames("[N,C,H,W]");
    auto parsedNames2 = Shape::parseNames(" [ N ,C,H,   W ]");
    auto realNames = std::vector<std::string> { "N", "C", "H", "W" };
    ASSERT_EQ(parsedNames1, realNames);
    ASSERT_EQ(parsedNames2, realNames);
}
