#include <Halide.h>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

class codegen_tests: public ::testing::Test {
protected:
    Sampler sampler = Sampler("[N,H,W]", "[N,H,W]", {"N=8", "H=16", "W=16"}, {"k=3", "s=2"}, {{}}, {{0, 0}});
    BindingContext& ctx = sampler.getBindingContext();
};

TEST_F(codegen_tests, generate) {
    auto [sizeN, sizeH, sizeW] = ctx.getSizes("N", "H", "W");
    Iterator itN { 0, sizeN }, itH { 1, sizeH }, itW { 2, sizeW };
    auto sample = TensorView({{ &itN, &itH, &itW }}, TensorExpression::ProductOfTensors(1));
    std::cout << sample.printNestedLoopsForAll(ctx);
    HalideGen gen { ctx, sample, HalideGen::Options() };
    auto consts = ctx.realizeConsts({});
    gen.generate("./kernel_codegen_test/forward", "./kernel_codegen_test/backward", "codegen_test_forward", "codegen_test_backward", consts, &std::cout);
}

} // namespace kas
