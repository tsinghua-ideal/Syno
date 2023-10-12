#include <Halide.h>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/CodeGen/PytorchGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

class codegen_tests: public ::testing::Test {
protected:
    Sampler sampler = Sampler("[N,H,W]", "[N,C]", {"N=8", "H=16", "W=16", "C=8"}, {"k=3", "s=2"}, {{}}, {{0, 0}});
    BindingContext& ctx = sampler.getBindingContext();
};

TEST_F(codegen_tests, halide_generate) {
    auto [sizeN, sizeH, sizeW] = ctx.getSizes("N", "H", "W");
    Iterator itN { 0, sizeN }, itH { 1, sizeH }, itW { 2, sizeW };
    auto sample = TensorView({{{&itN, &itH, &itW}, {}}}, TensorExpression::ProductOfTensors(1), ctx);
    std::cout << sample.printNestedLoopsForAll(ctx);
    HalideGen gen { ctx, sample, CodeGenOptions() };
    auto consts = ctx.realizeConsts({});
    gen.generate("./kernel_codegen_test/forward", "./kernel_codegen_test/backward", "codegen_test_forward", "codegen_test_backward", consts, &std::cout);
}

TEST_F(codegen_tests, torch_generate) {
    auto [sizeN, sizeH, sizeW, sizeC] = ctx.getSizes("N", "H", "W", "C");
    Iterator itN{0, sizeN}, itH{1, sizeH}, itW{2, sizeW}, itC{3, sizeC};
    auto sample = TensorView({{{&itN, &itH, &itW}, {}}},
                             TensorExpression::ProductOfTensors(1), ctx);
    std::cout << sample.printNestedLoopsForAll(ctx);
    PyTorchGen gen{ctx, sample};
    auto consts = ctx.realizeConsts({});
    gen.generateSingle("./kernel_codegen_test/torch_kernel.py", "torch_kernel",
                       sample, {});
}

} // namespace kas
