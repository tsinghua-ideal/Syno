#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Tensor.hpp"


using namespace kas;

TEST(core_tests, size) {
    std::vector<BindingContext::Metadata> metaPrimary { BindingContext::Metadata("H", 128), BindingContext::Metadata("W", 128) };
    std::vector<BindingContext::Metadata> metaCoefficient { BindingContext::Metadata("c", 5) };
    auto ctx = BindingContext { std::move(metaPrimary), std::move(metaCoefficient) };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    ASSERT_EQ(sizeH.toString(ctx), "H");
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    ASSERT_EQ(sizeW.toString(ctx), "W");
    auto sizeC = ctx.getSingleCoefficientVariableSize(0);
    auto sizeHWc = Size(2, 1, Size::ExprType { 1, 1 }, Size::ExprType { -1 });
    ASSERT_EQ(sizeH * sizeW / sizeC, sizeHWc);
    ASSERT_EQ(sizeHWc.toString(ctx), "c^-1*H*W");
    ASSERT_EQ((sizeH * sizeH).toString(ctx), "H^2");

    auto sizeOneOverC = sizeC.identity();
    ASSERT_EQ(sizeOneOverC.getTrait(), Size::Trait::One);
    ASSERT_EQ(LabeledSize { sizeC }.getTrait(), Size::Trait::Coefficient);
    auto trait = sizeOneOverC.testDividedBy(sizeC);
    ASSERT_EQ(trait.value(), Size::Trait::IllegalCoefficient);
    LabeledSize ls { sizeOneOverC };
    ASSERT_EQ(ls.getTrait(), Size::Trait::IllegalCoefficient);
    ASSERT_EQ((ls * LabeledSize { sizeH }).getTrait(), Size::Trait::General);
}

TEST(core_tests, tensor) {
    auto ctx = BindingContext { static_cast<std::size_t>(3), static_cast<std::size_t>(2) };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    ASSERT_EQ(sizeH.toString(ctx), "x_0");
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    ASSERT_EQ(sizeW.toString(ctx), "x_1");
    auto sizeC = ctx.getSingleCoefficientVariableSize(0);
    ASSERT_EQ(sizeC.toString(ctx), "c_0");
    auto shape = Shape { std::vector<Size> { sizeH, sizeW, sizeC } };
    Iterator i_0 { 0, sizeH }, i_1 { 1, sizeW }, i_2 { 2, sizeC };
    Interface interface { &i_0, &i_1, &i_2 };
    auto tensorView = TensorView { { interface } };
    ASSERT_EQ(tensorView.interfaceAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.shapeToString(ctx), "[x_0,x_1,c_0]");
    const auto& tensors = tensorView.getUnderlyingTensors();
    ASSERT_EQ(tensors.size(), 1);
    const auto& tensor = tensors[0];
    ASSERT_EQ(tensor.accessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensor.shapeToString(ctx), "[x_0,x_1,c_0]");
}

TEST(core_tests, parse_shape_names) {
    auto parsedNames1 = Size::parseNames("[N,C,H,W]");
    auto parsedNames2 = Size::parseNames(" [ N ,C,H,   W ]");
    auto realNames = std::vector<std::string> { "N", "C", "H", "W" };
    ASSERT_EQ(parsedNames1, realNames);
    ASSERT_EQ(parsedNames2, realNames);
}

TEST(core_tests, parse_specs) {
    auto spec1 = Parser("N").parseSizeSpec();
    auto spec1Expect = Parser::SizeSpec { .quantity = "N", .maxOccurrences = std::nullopt };
    ASSERT_EQ(spec1, spec1Expect);
    auto spec2 = Parser("N = 5").parseSizeSpec();
    auto spec2Expect = Parser::SizeSpec { .quantity = std::make_pair("N", 5), .maxOccurrences = std::nullopt };
    ASSERT_EQ(spec2, spec2Expect);
    auto spec3 = Parser("5").parseSizeSpec();
    auto spec3Expect = Parser::SizeSpec { .quantity = static_cast<std::size_t>(5), .maxOccurrences = std::nullopt };
    ASSERT_EQ(spec3, spec3Expect);
    auto spec4 = Parser("5: 10").parseSizeSpec();
    auto spec4Expect = Parser::SizeSpec { .quantity = static_cast<std::size_t>(5), .maxOccurrences = static_cast<std::size_t>(10) };
    ASSERT_EQ(spec4, spec4Expect);
}
