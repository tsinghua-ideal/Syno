#include <gtest/gtest.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

class codegen_tests: public ::testing::Test {
protected:
    Sampler sampler = { "[H,W]", "[N,C,H,W]", {} };
    BindingContext& ctx = sampler.getBindingContext();
};

TEST_F(codegen_tests, func) {
    auto sample = sampler.sample();
    HalideGen gen { ctx, sample };
    auto [params, func] = gen.createFunc("kernel_0");
    func.print_loop_nest();
}

TEST_F(codegen_tests, generate) {
    auto sample = sampler.sample();
    HalideGen gen { ctx, sample };
    gen.generate(".", "kernel_1");
}

} // namespace kas
