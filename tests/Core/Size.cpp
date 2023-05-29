#include <unordered_set>

#include <gtest/gtest.h>

#include "KAS/Core/Size.hpp"


namespace kas {

class core_size_tests: public ::testing::Test {
protected:
    using Metadata = BindingContext::Metadata;
    std::vector<Metadata> metaPrimary {
        { .alias = "H", .maximumOccurrence = 2, .estimate = 128 },
        { .alias = "W", .maximumOccurrence = 2, .estimate = 128 },
    };
    std::vector<Metadata> metaCoefficient {
        { .alias = "c", .maximumOccurrence = 2, .estimate = 5 },
    };
    BindingContext ctx = { std::move(metaPrimary), std::move(metaCoefficient) };
    Size sizeH = ctx.getSize("H");
    Size sizeW = ctx.getSize("W");
    Size sizeC = ctx.getSize("c");
    Size sizeHWc = Size(2, 1, Size::ExprType { 1, 1 }, Size::ExprType { -1 });
};

TEST_F(core_size_tests, arithmetics) {
    ASSERT_EQ(sizeH.toString(ctx), "H");
    ASSERT_EQ(sizeW.toString(ctx), "W");
    ASSERT_EQ(sizeH * sizeW / sizeC, sizeHWc);
    ASSERT_EQ(sizeHWc.toString(ctx), "c^-1*H*W");
    ASSERT_EQ((sizeH * sizeH).toString(ctx), "H^2");
}

TEST_F(core_size_tests, trait) {
    auto sizeOneOverC = sizeC.identity();
    ASSERT_EQ(sizeOneOverC.getTrait(), Size::Trait::One);
    ASSERT_EQ(LabeledSize { sizeC }.getTrait(), Size::Trait::Coefficient);
    auto trait = sizeOneOverC.testDividedBy(sizeC);
    ASSERT_EQ(trait.value(), Size::Trait::IllegalCoefficient);
    LabeledSize ls { sizeOneOverC };
    ASSERT_EQ(ls.getTrait(), Size::Trait::IllegalCoefficient);
    ASSERT_EQ((ls * LabeledSize { sizeH }).getTrait(), Size::Trait::General);
}

TEST_F(core_size_tests, divisors_HWoverC) {
    std::unordered_set<Size> divisors;
    std::ranges::move(sizeHWc.sampleDivisors(ctx), std::inserter(divisors, divisors.end()));
    ASSERT_TRUE(divisors.contains(sizeH));
}

TEST_F(core_size_tests, divisors_H) {
    Size query = sizeH;
    fmt::print("Divisors of {}:\n", query.toString(ctx));
    for (auto divisor: query.sampleDivisors(ctx)) {
        fmt::print("  {}\n", divisor.toString(ctx));
    }
}

TEST_F(core_size_tests, divisors_HWC3) {
    Size query = sizeH * sizeW * sizeC * sizeC * sizeC;
    fmt::print("Divisors of {}:\n", query.toString(ctx));
    for (auto divisor: query.sampleDivisors(ctx)) {
        fmt::print("  {}\n", divisor.toString(ctx));
    }
}

TEST_F(core_size_tests, allowance) {
    // Maximum occurences: H -> 2, W -> 2, c -> 2
    Allowance allowance = { sizeHWc, ctx };
    // Allowed: H -> [0, 1], W -> [0, 1], c -> [-1, 3]
    ASSERT_TRUE(allowance.withinAllowance(sizeH / sizeC));
    ASSERT_TRUE(allowance.withinAllowance(sizeH * sizeW * sizeC * sizeC * sizeC));
    ASSERT_FALSE(allowance.withinAllowance(sizeH * sizeW * sizeC * sizeC * sizeC * sizeC));
    ASSERT_FALSE(allowance.withinAllowance(sizeH * sizeH));
}

TEST_F(core_size_tests, enumerate_HWC) {
    Size query = sizeH * sizeW * sizeC;
    fmt::print("Enumerate {}:\n", query.toString(ctx));
    std::size_t counter = std::ranges::distance(Size::EnumerateSizes(ctx, query.identity(), query));
    ASSERT_EQ(counter, 8 - 1);
}

TEST_F(core_size_tests, enumerate_HWoverC) {
    auto lower = sizeHWc.identity() / sizeC / sizeC;
    fmt::print("Enumerate {} ~ {}:\n", lower.toString(ctx), sizeHWc.toString(ctx));
    for (auto size: Size::EnumerateSizes(ctx, lower, sizeHWc)) {
        fmt::print("  {}\n", size.toString(ctx));
    }
}

TEST_F(core_size_tests, leq) {
    ASSERT_TRUE(
        Size::LexicographicalLEQ(sizeC, sizeH)
        != Size::LexicographicalLEQ(sizeH, sizeC)
    );
    ASSERT_TRUE(Size::LexicographicalLEQ(sizeH, sizeH));
    ASSERT_TRUE(
        Size::LexicographicalLEQ(sizeH, sizeH * sizeC)
        != Size::LexicographicalLEQ(sizeH * sizeC, sizeH)
    );
}

} // namespace kas
