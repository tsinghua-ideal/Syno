#include <fmt/format.h>
#include <gtest/gtest.h>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/CodeGen/PyTorchGen.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Statistics.hpp"
#include "KAS/Transforms/Forward.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

class search_tests: public ::testing::Test {
protected:
    int dimC_in = 3, dimC_out = 8, dimH = 64, dimW = 64, dimK1 = 3, dimS1 = 2;
    bool doRealization = false;
    std::map<std::string, std::size_t> dict { { "C_in", dimC_in }, { "C_out", dimC_out }, { "H", dimH }, { "W", dimW }, { "k_1", dimK1 }, { "s_1", dimS1 } };
    SampleOptions options = []() {
        SampleOptions options;
        options.seed = 42;
        options.depth = 12;
        options.maximumTensors = 3;
        options.maximumReductions = 3;
        options.maxFLOPs = 5e6;
        options.maxChainLength = 5;
        options.minSingleWeightParams = 1;
        return options;
    }();
    Sampler sampler = {"[N,C_in:unordered,H,W]", "[N,C_out:unordered,H,W]", {"N=3:0", "C_in:2", "C_out:2", "H:0", "W:0"}, {"k_1=3:4", "s_1=2:0"}, {dict}, {{0, 0}}, options, 12};
    const BindingContext& ctx = sampler.getBindingContext();
    search_tests() {
        ctx.debug(); // For debugging.
    }
};

} // namespace kas
