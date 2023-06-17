#include "Prelude.hpp"


namespace kas {

TEST_F(search_tests, sampler) {
    constexpr std::size_t trials = 500;
    std::size_t successes = 0;
    for (int i = 0; i < trials; ++i) {
        auto randomLeaf = sampler.randomNodeWithPrefix({});
        if (!randomLeaf || !randomLeaf->second.isFinal()) {
            fmt::print("Trial {} failed.\n", i);
            continue;
        } else {
            fmt::print("Trial {} succeeded.\n", i);
        }
        auto [_, node] = *randomLeaf;
        ++successes;
        auto& tensorView = *node.asFinal();

        auto r = tensorView.getUnderlyingTensors() | std::ranges::views::transform([&](const auto& tensor) { return tensor.shapeToString(ctx); });
        std::cout << fmt::format("Input Shape: {}", fmt::join(r, ", ")) << std::endl;
        std::cout << tensorView.printNestedLoopsForAll(ctx);

        GraphvizGen(tensorView, ctx).generate("./search_viz", "trial_" + std::to_string(i));

        if (doRealization) {
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
