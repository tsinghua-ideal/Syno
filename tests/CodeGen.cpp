#include <gtest/gtest.h>
#include <iostream>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

class codegen_tests: public ::testing::Test {
protected:
    Sampler sampler = { "[H,W]", "[N,C,H,W]", SampleOptions() };
    BindingContext& ctx = sampler.getBindingContext();
};

TEST_F(codegen_tests, func) {
    auto sample = std::get<TensorView>(sampler.randomSample());
    HalideGen gen { ctx, sample };
    auto [params, func] = gen.createFunc("kernel_0");
    func.print_loop_nest();
}

TEST_F(codegen_tests, generate) {
    auto sample = std::get<TensorView>(sampler.randomSample());
    std::cout << sample.printNestedLoops(ctx) << std::endl;
    HalideGen gen { ctx, sample };
    gen.generate("./kernel_1", "kernel_1", {
        .useGPU = false,
        .scheduler = HalideGen::Options::AutoScheduler::Li2018
    });
}

TEST_F(codegen_tests, path) {
    ASSERT_EQ(sampler.isFinal({ 0 }), true);
    auto [tensorView, cgCtx, repr] = sampler.realize({ 0 });
    HalideGen gen { ctx, tensorView };
    gen.generate("./kernel_2", "kernel_2", {
        .useGPU = false,
        .scheduler = HalideGen::Options::AutoScheduler::Li2018
    });
}

} // namespace kas
