#include <gtest/gtest.h>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

TEST(codegen_tests, func) {
    Sampler sampler("[H,W]", "[N,C,H,W]", SampleOptions());
    auto& ctx = sampler.getBindingContext();
    auto sample = sampler.sample();
    HalideGen gen(ctx, sample);
    auto [params, func] = gen.createFunc();
    func.print_loop_nest();
}

} // namespace kas
