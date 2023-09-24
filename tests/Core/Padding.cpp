#include <gtest/gtest.h>

#include "KAS/Core/Size.hpp"


using namespace kas;

class core_padding_tests: public ::testing::Test {
protected:
    BindingContext ctx = BindingContext({"C=3", "H=128", "W=40"}, {"k=5", "s=2"});
    Size sizeC = ctx.getSize("C");
    Size sizeH = ctx.getSize("H");
    Size sizeW = ctx.getSize("W");
    Size sizeK = ctx.getSize("k");
    Size sizeS = ctx.getSize("s");
    ConcreteConsts unpaddedConsts = ctx.realizeConsts({});
    PaddingSolver sol { ctx, unpaddedConsts };
};

TEST_F(core_padding_tests, none) {
    sol.addConstraint(sizeH * sizeW / sizeS);
    sol.addConstraint(sizeH / (sizeS * sizeS));
    sol.addConstraint(sizeW / (sizeS * sizeS * sizeS));
    auto padded = sol.solve(sizeC * sizeH * sizeW, sizeC * sizeH * sizeW);
    EXPECT_EQ(padded, unpaddedConsts);
}

TEST_F(core_padding_tests, one) {
    sol.addConstraint(sizeC / sizeK);
    auto padded = sol.solve(sizeC * sizeH * sizeW, sizeC * sizeH * sizeW);
    EXPECT_EQ(padded, (ConcreteConsts{
        .primary = { 5, 128, 40 },
        .coefficient = { 5, 2 },
    }));
}

TEST_F(core_padding_tests, two) {
    sol.addConstraint(sizeC * sizeH / sizeK);
    sol.addConstraint(sizeC * sizeW / (sizeS * sizeS * sizeS * sizeS));
    auto padded = sol.solve(sizeC * sizeH * sizeW, sizeC * sizeH * sizeW);
    EXPECT_EQ(padded, (ConcreteConsts{
        .primary = { 3, 130, 48 },
        .coefficient = { 5, 2 },
    }));
}
