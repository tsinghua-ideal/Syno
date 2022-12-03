#include <gtest/gtest.h>
#include <KAS/Core/Shape.hpp>
#include <vector>


using namespace kas;

TEST(core_tests, size) {
    std::vector<BindingContext::Metadata> metaPrimary { BindingContext::Metadata("H"), BindingContext::Metadata("W") };
    auto ctx = BindingContext { std::move(metaPrimary), std::vector<BindingContext::Metadata>() };
    auto sizeH = ctx.getSinglePrimaryVariableSize(0);
    ASSERT_EQ(sizeH->toString(ctx), "H");
    auto sizeW = ctx.getSinglePrimaryVariableSize(1);
    ASSERT_EQ(sizeW->toString(ctx), "W");
    auto sizeHW = Size { std::vector<int> { 1, 1 }, std::vector<int>() };
    ASSERT_EQ(*sizeH * *sizeW, sizeHW);
    ASSERT_EQ(sizeHW.toString(ctx), "HW");
}
