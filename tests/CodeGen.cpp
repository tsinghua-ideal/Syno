#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

class codegen_tests: public ::testing::Test {
protected:
    Sampler sampler = { "[H,W]", "[N,C,H,W]", {}, {"k", "s"}, SampleOptions {
        .seed = 19216811,
        .depth = 3,
        .dimLowerBound = 1,
        .dimUpperBound = 8,
    } };
    BindingContext& ctx = sampler.getBindingContext();
};

TEST_F(codegen_tests, func) {
    auto sample = std::get<TensorView>(sampler.randomSample());
    HalideGen gen { ctx, sample };
    auto [params, func] = gen.createFunc("kernel_0");
    func.print_loop_nest();
}

TEST_F(codegen_tests, generate) {
    for (int i = 0; i < 3; ++i) {
        auto [sample, _] = sampler.randomSample();
        std::cout << sample.printNestedLoops(ctx);
        HalideGen gen { ctx, sample };
        gen.generate("./kernel_1_" + std::to_string(i), "kernel_1_" + std::to_string(i), {
            .useGPU = false,
            .scheduler = HalideGen::Options::AutoScheduler::Mullapudi2016
        });
    }
}

TEST_F(codegen_tests, path) {
    ASSERT_EQ(sampler.isFinal({ 0 }), true);
    auto [tensorView, cgCtx] = sampler.realize({ 0 });
    HalideGen gen { ctx, tensorView };
    gen.generate("./kernel_2", "kernel_2", {
        .useGPU = false,
        .scheduler = HalideGen::Options::AutoScheduler::Li2018
    });
}

} // namespace kas
