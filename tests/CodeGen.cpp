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
    Sampler sampler = Sampler("[N,H,W]", "[N,H,W]", {"N=8", "H=16", "W=16"}, {"k=5", "s=2"}, {{}}, {{0, 0}});
    BindingContext& ctx = sampler.getBindingContext();
};

TEST_F(codegen_tests, generate) {
    std::size_t i = 0;
    while (true) {
        auto [_, node] = sampler.randomNodeWithPrefix({});
        if (!node.isFinal()) {
            ++i;
            continue;
        }
        auto sample = node.asFinal();
        std::cout << sample->printNestedLoopsForAll(ctx);
        HalideGen gen { ctx, *sample, HalideGen::Options() };
        auto consts = ctx.realizeConsts({});
        gen.generate("./kernel_1_" + std::to_string(i), "kernel_1_" + std::to_string(i), consts);
        break;
    }
}

} // namespace kas
