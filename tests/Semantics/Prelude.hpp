#include <fstream>
#include <random>

#include <gtest/gtest.h>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/CodeGen/PyTorchGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Transforms/Forward.hpp"


namespace kas {

class semantics_tests: public ::testing::Test {
protected:
    using Mappings = std::map<std::string, std::size_t>;
    const CodeGenOptions options = {
        .halide = true,
        .useGPU = true,
        .scheduler = CodeGenOptions::AutoScheduler::Anderson2021,
        .extraOptions = {
            {"beam_size", "32"},
            {"num_passes", "1"},
            {"search_space_options", "1000"},
            {"parallelism", "30"},
            {"shared_memory_limit_kb", "48"},
            {"shared_memory_sm_limit_kb", "64"},
            {"active_block_limit", "512"},
            {"active_warp_limit", "1024"},
        },
        .rfactorThreshold = 32,
        .inBoundsLikelyThreshold = 0.3,
    };
    const bool doSemanticTests = true;
    const bool createStaticLibrary = true;
    std::mt19937 rng { std::random_device()() };
    std::uniform_real_distribution<float> dist { 0.5f, 1.5f };
    float random() { return dist(rng); }
};

} // namespace kas
