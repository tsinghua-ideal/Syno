#include <gtest/gtest.h>

#include "KAS/Core/Size.hpp"


using namespace kas;

TEST(core_tests, size) {
    using Metadata = BindingContext::Metadata;
    std::vector<Metadata> metaPrimary {
        { .alias = "H", .estimate = 128 },
        { .alias = "W", .estimate = 128 },
    };
    std::vector<Metadata> metaCoefficient {
        { .alias = "c", .estimate = 5 },
    };
    auto ctx = BindingContext { std::move(metaPrimary), std::move(metaCoefficient) };
    auto sizeH = ctx.get("H");
    ASSERT_EQ(sizeH.toString(ctx), "H");
    auto sizeW = ctx.get("W");
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
