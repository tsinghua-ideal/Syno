#include <gtest/gtest.h>

#include "KAS/Core/Size.hpp"


using namespace kas;

class padding_tests: public ::testing::Test {
protected:
    using Metadata = BindingContext::Metadata;
    BindingContext ctx =  {
        {
            { .alias = "C", .estimate = 3 },
            { .alias = "H", .estimate = 128 },
            { .alias = "W", .estimate = 40 },
        },
        {
            { .alias = "k", .estimate = 5 },
            { .alias = "s", .estimate = 2 },
        },
    };
    Size sizeC = ctx.get("C");
    Size sizeH = ctx.get("H");
    Size sizeW = ctx.get("W");
    Size sizeK = ctx.get("k");
    Size sizeS = ctx.get("s");
    ConcreteConsts unpaddedConsts = ctx.realizeConsts({});
    PaddingSolver sol { ctx, unpaddedConsts };
};

TEST_F(padding_tests, none) {
    sol.addConstraint(sizeH * sizeW / sizeS);
    sol.addConstraint(sizeH / (sizeS * sizeS));
    sol.addConstraint(sizeW / (sizeS * sizeS * sizeS));
    auto padded = sol.solve(sizeC * sizeH * sizeW, sizeC * sizeH * sizeW);
    EXPECT_EQ(padded, unpaddedConsts);
}

TEST_F(padding_tests, one) {
    sol.addConstraint(sizeC / sizeK);
    auto padded = sol.solve(sizeC * sizeH * sizeW, sizeC * sizeH * sizeW);
    EXPECT_EQ(padded, (ConcreteConsts{
        .primary = { 5, 128, 40 },
        .coefficient = { 5, 2 },
    }));
}

TEST_F(padding_tests, two) {
    sol.addConstraint(sizeC * sizeH / sizeK);
    sol.addConstraint(sizeC * sizeW / (sizeS * sizeS * sizeS * sizeS));
    auto padded = sol.solve(sizeC * sizeH * sizeW, sizeC * sizeH * sizeW);
    EXPECT_EQ(padded, (ConcreteConsts{
        .primary = { 3, 130, 48 },
        .coefficient = { 5, 2 },
    }));
}
