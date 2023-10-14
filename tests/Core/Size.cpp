#include <unordered_set>

#include <gtest/gtest.h>

#include "KAS/Core/Size.hpp"


namespace kas {

class core_size_tests: public ::testing::Test {
protected:
    BindingContext ctx = BindingContext({"H=128:2", "W=128:2"}, {"c=5:2"});
    Size sizeH = ctx.getSize("H");
    Size sizeW = ctx.getSize("W");
    Size sizeC = ctx.getSize("c");
    Size sizeHWc = ctx.getSize("c^-1*H*W");
    Allowance allowance { ctx, Size::Identity(ctx), true };
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
    auto trait = sizeOneOverC.testDividedBy(sizeC);
    ASSERT_EQ(trait.value(), Size::Trait::IllegalCoefficient);
}

TEST_F(core_size_tests, divisors_HWoverC) {
    std::unordered_set<Size> divisors;
    std::ranges::move(allowance.enumerateDivisors(sizeHWc), std::inserter(divisors, divisors.end()));
    ASSERT_TRUE(divisors.contains(sizeH));
}

TEST_F(core_size_tests, enumerate_sizes) {
    std::unordered_set<Size> sizes;
    std::ranges::move(allowance.enumerateSizes(), std::inserter(sizes, sizes.end()));
    ASSERT_TRUE(sizes.contains(sizeH));
    ASSERT_TRUE(sizes.contains(sizeW));
    ASSERT_TRUE(sizes.contains(sizeC));
    ASSERT_TRUE(sizes.contains(sizeHWc));
    ASSERT_FALSE(sizes.contains(ctx.getSize("c^3")));
    ASSERT_FALSE(sizes.contains(ctx.getSize("c^-3*H*W")));
    ASSERT_TRUE(sizes.contains(ctx.getSize("c^-2*H*W")));
}

TEST_F(core_size_tests, divisors_H) {
    Size query = sizeH;
    fmt::print("Divisors of {}:\n", query.toString(ctx));
    for (auto divisor: allowance.enumerateDivisors(query)) {
        fmt::print("  {}\n", divisor.toString(ctx));
    }
}

TEST_F(core_size_tests, divisors_HWC3) {
    Size query = sizeH * sizeW * sizeC * sizeC * sizeC;
    fmt::print("Divisors of {}:\n", query.toString(ctx));
    for (auto divisor: allowance.enumerateDivisors(query)) {
        fmt::print("  {}\n", divisor.toString(ctx));
    }
}

TEST(core_size_tests_standalone, divisors_W) {
    BindingContext ctx = BindingContext({"W=64:1"}, {"s=2:4", "g=32:4"});
    auto [sizeW_over_s, sizeG] = ctx.getSizes("s^-1*W", "g");
    Allowance allowance { ctx, Size::Identity(ctx), true };
    std::unordered_set<Size> divisors;
    std::ranges::move(allowance.enumerateDivisors(sizeW_over_s), std::inserter(divisors, divisors.end()));
    ASSERT_TRUE(divisors.contains(sizeG));
}

TEST_F(core_size_tests, allowance) {
    // Maximum occurences: H -> 2, W -> 2, c -> 2
    Allowance allowance = { ctx, sizeH * sizeW * sizeC, true };
    // Allowed: H -> [0, 1], W -> [0, 1], c -> [-1, 1]
    ASSERT_TRUE(allowance.shareWithinAllowance(sizeH / sizeC));
    ASSERT_TRUE(allowance.shareWithinAllowance(sizeH * sizeW * sizeC));
    ASSERT_FALSE(allowance.shareWithinAllowance(sizeH * sizeW * sizeC * sizeC));
    ASSERT_FALSE(allowance.shareWithinAllowance(sizeH * sizeH));
}

TEST_F(core_size_tests, enumerate_HWC) {
    Size query = sizeH * sizeW * sizeC;
    fmt::print("Enumerate {}:\n", query.toString(ctx));
    std::size_t counter = std::ranges::distance(Size::EnumerateSizes(ctx, Size::Identity(ctx), query));
    ASSERT_EQ(counter, 8 - 1);
}

TEST_F(core_size_tests, enumerate_HWoverC) {
    auto lower = Size::Identity(ctx) / sizeC / sizeC;
    fmt::print("Enumerate {} ~ {}:\n", lower.toString(ctx), sizeHWc.toString(ctx));
    std::size_t counter = 0;
    for (auto size: Size::EnumerateSizes(ctx, lower, sizeHWc)) {
        fmt::print("  {}\n", size.toString(ctx));
        ++counter;
    }
    ASSERT_EQ(counter, 6);
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
