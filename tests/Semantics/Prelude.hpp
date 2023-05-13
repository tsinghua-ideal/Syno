#include <random>

#include <gtest/gtest.h>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Transforms/Forward.hpp"


namespace kas {

class semantics_tests: public ::testing::Test {
protected:
    using SizeName = BindingContext::Metadata;
    using Mappings = std::map<std::string, std::size_t>;
    const HalideGen::Options options = {
        .useGPU = true,
        .scheduler = HalideGen::Options::AutoScheduler::Anderson2021,
    };
    const bool doSemanticTests = true;
    const bool createStaticLibrary = true;
    std::mt19937 rng { std::random_device()() };
    std::uniform_real_distribution<float> dist { 0.5f, 1.5f };
    float random() { return dist(rng); }
};

} // namespace kas
