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
    Sampler sampler = { "[H,W]", "[N,C,H,W]", {"N=8", "C=3", "H=16", "W=16"}, {"k=5", "s=2"}, SampleOptions {
        .seed = 19216811,
        .depth = 3,
        .dimLowerBound = 1,
        .dimUpperBound = 8,
    } };
    BindingContext& ctx = sampler.getBindingContext();
};

TEST_F(codegen_tests, generate) {
    std::size_t i = 0;
    while (true) {
        auto node = sampler.randomNodeWithPrefix({});
        if (!node.isFinal()) {
            ++i;
            continue;
        }
        auto sample = node.asKernel();
        std::cout << sample->printNestedLoopsForAll(ctx);
        HalideGen gen { ctx, *sample, {
            .useGPU = false,
            .scheduler = HalideGen::Options::AutoScheduler::ComputeRoot,
        } };
        gen.generate("./kernel_1_" + std::to_string(i), "kernel_1_" + std::to_string(i), {});
        break;
    }
}

} // namespace kas
