#include <gtest/gtest.h>

#include "KAS/Core/Parser.hpp"
#include "KAS/Core/Size.hpp"


using namespace kas;

TEST(core_parser_tests, parse_shape_names) {
    auto parsedNames1 = Size::parseNames("[N,C,H,W]");
    auto parsedNames2 = Size::parseNames(" [ N ,C,H,   W ]");
    auto realNames = std::vector<std::string> { "N", "C", "H", "W" };
    ASSERT_EQ(parsedNames1, realNames);
    ASSERT_EQ(parsedNames2, realNames);
}

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
