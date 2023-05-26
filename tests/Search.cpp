#include <map>
#include <memory>
#include <ranges>
#include <string>
#include <vector>

#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Statistics.hpp"
#include "KAS/Transforms.hpp"


namespace kas {

TEST(search_tests, sampler) {
    constexpr int dimH = 64, dimW = 64, dimK1 = 3, dimS1 = 2;
    constexpr bool doRealization = false;
    std::map<std::string, std::size_t> dict { { "H", dimH }, { "W", dimW }, { "k_1", dimK1 }, { "s_1", dimS1 } };

    SampleOptions options;
    options.seed = 42;
    options.depth = 10;
    options.dimLowerBound = 2;
    options.dimUpperBound = 6;
    options.maximumTensors = 2;
    options.maxFLOPs = 1e6;
    Sampler sampler("[N,H,W]", "[N,H,W]", {"N=3:0"}, {"k_1=3", "s_1=2"}, {dict}, {{0, 0}}, options);
    auto& ctx = sampler.getBindingContext();
    BindingContext::DebugPublicCtx = &ctx; // For debugging.
    ASSERT_EQ(ctx.getPrimaryCount(), 3);
    ASSERT_EQ(ctx.getCoefficientCount(), 2);

    constexpr std::size_t trials = 100;
    std::size_t successes = 0;
    for (int i = 0; i < trials; ++i) {
        auto [_, node] = sampler.randomNodeWithPrefix({});
        if (!node.isFinal()) {
            fmt::print("Trial {} failed.\n", i);
            continue;
        } else {
            fmt::print("Trial {} succeeded.\n", i);
        }
        ++successes;
        auto& tensorView = *node.asFinal();

        auto r = tensorView.getUnderlyingTensors() | std::ranges::views::transform([&](const auto& tensor) { return tensor.shapeToString(ctx); });
        std::cout << fmt::format("Input Shape: {}", fmt::join(r, ", ")) << std::endl;
        std::cout << tensorView.printNestedLoopsForAll(ctx);

        GraphvizGen(tensorView, ctx).generate("./search_viz", "trial_" + std::to_string(i));

        if constexpr (doRealization) {
            HalideGen gen(ctx, tensorView, HalideGen::Options());
            auto name = "search_codegen_test_" + std::to_string(i);
            try {
                gen.performTrial<false>(dict, name, true, false, []{});
            } catch (const Halide::Error& e) {
                fmt::print("Trial {} failed at runtime: {}\n", i, e.what());
            }
        }
    }
    StatisticsCollector::PrintSummary(std::cout);
    fmt::print("Success rate: {:.2f} ({} / {})\n", static_cast<float>(successes) / trials, successes, trials);
}

} // namespace kas
