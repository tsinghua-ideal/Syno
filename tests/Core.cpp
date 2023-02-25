#include <cstddef>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/MapReduce.hpp"
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
    auto sizeX0 = ctx.getSinglePrimaryVariableSize(0);
    ASSERT_EQ(sizeX0.toString(ctx), "x_0");
    auto sizeX1 = ctx.getSinglePrimaryVariableSize(1);
    ASSERT_EQ(sizeX1.toString(ctx), "x_1");
    auto sizeC0 = ctx.getSingleCoefficientVariableSize(0);
    ASSERT_EQ(sizeC0.toString(ctx), "c_0");
    auto sizeC1 = ctx.getSingleCoefficientVariableSize(1);
    ASSERT_EQ(sizeC1.toString(ctx), "c_1");
    Iterator i_0 { 0, sizeX0 }, i_1 { 1, sizeX1 }, i_2 { 2, sizeC0 };
    MapReduceOp i_3 { 0, sizeC1, MapReduceOp::MapType::Identity, MapReduceOp::ReduceType::Sum };
    Interface interface { &i_0, &i_1, &i_2, &i_3 };
    auto tensorView = TensorView { { interface } };
    ASSERT_EQ(tensorView.getShape().toString(ctx), "[x_0,x_1,c_0]");
    ASSERT_EQ(tensorView.getReduceShape().toString(ctx), "[c_1]");
    ASSERT_EQ(tensorView.interfaceAccessToString(ctx), "[i_0,i_1,i_2]");
    ASSERT_EQ(tensorView.reduceAccessToString(ctx), "[ri_0]");
    ASSERT_EQ(tensorView.actualAccessToString(ctx), "[i_0,i_1,i_2,ri_0] with ri_0 Sum reduced");
    ASSERT_EQ(tensorView.printNestedLoops(ctx, "out"),
R"(for (int i_0 = 0; i_0 < x_0; i_0++) {
    for (int i_1 = 0; i_1 < x_1; i_1++) {
        for (int i_2 = 0; i_2 < c_0; i_2++) {
            float temp_ri_0 = 0;
            for (int ri_0 = 0; ri_0 < c_1; ri_0++) {
                temp_ri_0 += in_0[i_0,i_1,i_2,ri_0];
            }
            out[i_0,i_1,i_2] = temp_ri_0;
        }
    }
}
)");
    const auto& tensors = tensorView.getUnderlyingTensors();
    ASSERT_EQ(tensors.size(), 1);
    const auto& tensor = tensors[0];
    ASSERT_EQ(tensor.accessToString(ctx), "[i_0,i_1,i_2,ri_0]");
    ASSERT_EQ(tensor.shapeToString(ctx), "[x_0,x_1,c_0,c_1]");
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
