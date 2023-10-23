#include <gtest/gtest.h>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/Size.hpp"


using namespace kas;

TEST(core_parser_tests, parse_specs) {
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

TEST(core_parser_tests, parse_size) {
    auto size1 = Parser("x_1 ^ 2   *x_2* x_3^ -1").parseSize();
    auto size1Expect = std::vector<Parser::Factor> {{"x_1", 2}, {"x_2", 1}, {"x_3", -1}};
    ASSERT_EQ(size1, size1Expect);
    BindingContext ctx(4, 0);
    ASSERT_EQ(ctx.getSize("x_1 ^ 2   *x_2* x_3^ -1").toString(ctx), "x_1^2*x_2*x_3^-1");
}

TEST(core_parser_tests, parse_tensor_expression) {
    auto expr1 = Parser("in_0 * in_1 + in_2").parseTensorExpression();
    ASSERT_EQ(expr1.toString(), "in_0 * in_1 + in_2");
}
