#include <fmt/format.h>
#include <gtest/gtest.h>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Statistics.hpp"
#include "KAS/Transforms/Forward.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

class search_tests: public ::testing::Test {
protected:
    int dimH = 64, dimW = 64, dimK1 = 3, dimS1 = 2;
    bool doRealization = false;
    std::map<std::string, std::size_t> dict { { "H", dimH }, { "W", dimW }, { "k_1", dimK1 }, { "s_1", dimS1 } };
    SampleOptions options = []() {
        SampleOptions options;
        options.seed = 42;
        options.depth = 6;
        options.dimLowerBound = 2;
        options.dimUpperBound = 6;
        options.maximumTensors = 3;
        options.maximumReductions = 2;
        options.maxFLOPs = 1e7;
        return options;
    }();
    Sampler sampler = {"[N,H,W]", "[N,H,W]", {"N=3:0"}, {"k_1=3:4", "s_1=2"}, {dict}, {{0, 0}}, options, 12};
    const BindingContext& ctx = sampler.getBindingContext();
    search_tests() {
        BindingContext::DebugPublicCtx = &ctx; // For debugging.
    }
};

} // namespace kas
