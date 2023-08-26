#include "Prelude.hpp"


namespace kas {

TEST_F(search_tests, sampler) {
    constexpr std::size_t trials = 500;
    std::size_t successes = 0;
    std::size_t successfulReconstruction = 0;
    std::size_t failedReconstruction = 0;
    for (int i = 0; i < trials; ++i) {
        auto randomLeaves = sampler.randomFinalNodesWithPrefix({}, 32);
        for (const auto& [_, node]: randomLeaves) {
            auto path = sampler.convertTensorViewToPath(*node.asFinal());
            auto reconstructedNode = sampler.visit(path);
            if (reconstructedNode.has_value()) {
                ASSERT_EQ(
                    node.asFinal()->printNestedLoopsForAll(ctx),
                    reconstructedNode->asFinal()->printNestedLoopsForAll(ctx)
                );
                ++successfulReconstruction;
            } else {
                ++failedReconstruction;
            }
        }
        if (randomLeaves.empty() || !randomLeaves[0].second.isFinal()) {
            fmt::print("Trial {} failed.\n", i);
            continue;
        } else {
            fmt::print("Trial {} succeeded.\n", i);
        }
        auto [_, node] = randomLeaves[0];
        ++successes;
        auto& tensorView = *node.asFinal();

        auto r = tensorView.getUnderlyingTensors() | std::ranges::views::transform([&](const auto& tensor) { return tensor.shapeToString(ctx); });
        std::cout << fmt::format("Input Shape: {}", fmt::join(r, ", ")) << std::endl;
        std::cout << tensorView.printNestedLoopsForAll(ctx);

        PyTorchGen(ctx, tensorView).generateSingle("./search_pt/trial_" + std::to_string(i) + ".py", "trial_" + std::to_string(i), tensorView, dict);
        GraphvizGen(tensorView, ctx).generate("./search_viz/trial_" + std::to_string(i) + ".dot", "trial_" + std::to_string(i));

        if (doRealization) {
            auto cgOpt = CodeGenOptions();
            cgOpt.scheduler = CodeGenOptions::AutoScheduler::Anderson2021;
            cgOpt.useGPU = true;
            HalideGen gen(ctx, tensorView, cgOpt);
            auto name = "search_codegen_test_" + std::to_string(i);
            gen.performTrial<false>(dict, name, true, false, []{});
        }
    }
    StatisticsCollector::PrintSummary(std::cout);
    fmt::print("Success rate: {:.2f} ({} / {})\n", static_cast<float>(successes) / trials, successes, trials);
    fmt::print("Reconstruction: successful = {}, failed = {}\n", successfulReconstruction, failedReconstruction);
    ASSERT_GT(successfulReconstruction, failedReconstruction);
}

TEST_F(search_tests, sampler_2step) {
    constexpr std::size_t trials = 500;
    std::size_t successes = 0;
    std::size_t successfulReconstruction = 0;
    std::size_t failedReconstruction = 0;
    for (int i = 0; i < trials; ++i) {
        auto randomLeaves = sampler.randomFinalNodesWithPrefix({}, 32, Next::Type::Merge, 2);
        for (const auto &[pathSampled, node] : randomLeaves) {
            auto path = sampler.convertTensorViewToPath(*node.asFinal());
            auto reconstructedNode = sampler.visit(path);
            ASSERT_EQ(pathSampled[0].type, Next::Type::Merge);
            if (reconstructedNode.has_value()) {
                ASSERT_EQ(
                    node.asFinal()->printNestedLoopsForAll(ctx),
                    reconstructedNode->asFinal()->printNestedLoopsForAll(ctx));
                ++successfulReconstruction;
            } else {
                ++failedReconstruction;
            }
        }
        if (randomLeaves.empty() || !randomLeaves[0].second.isFinal()) {
            fmt::print("Trial {} failed.\n", i);
            continue;
        } else {
            fmt::print("Trial {} succeeded.\n", i);
        }
        auto [_, node] = randomLeaves[0];
        ++successes;
        auto &tensorView = *node.asFinal();

        auto r = tensorView.getUnderlyingTensors() |
                 std::ranges::views::transform([&](const auto &tensor) {
                   return tensor.shapeToString(ctx);
                 });
        std::cout << fmt::format("Input Shape: {}", fmt::join(r, ", "))
                  << std::endl;
        std::cout << tensorView.printNestedLoopsForAll(ctx);

        PyTorchGen(ctx, tensorView)
            .generateSingle("./search_pt/trial_" + std::to_string(i) + ".py",
                            "trial_" + std::to_string(i), tensorView, dict);
        GraphvizGen(tensorView, ctx)
            .generate("./search_viz/trial_" + std::to_string(i) + ".dot",
                      "trial_" + std::to_string(i));

        if (doRealization) {
            auto cgOpt = CodeGenOptions();
            cgOpt.scheduler = CodeGenOptions::AutoScheduler::Anderson2021;
            cgOpt.useGPU = true;
            HalideGen gen(ctx, tensorView, cgOpt);
            auto name = "search_codegen_test_" + std::to_string(i);
            gen.performTrial<false>(dict, name, true, false, [] {});
        }
    }
    StatisticsCollector::PrintSummary(std::cout);
    fmt::print("Success rate: {:.2f} ({} / {})\n",
               static_cast<float>(successes) / trials, successes, trials);
    fmt::print("Reconstruction: successful = {}, failed = {}\n",
               successfulReconstruction, failedReconstruction);
    ASSERT_GT(successfulReconstruction, failedReconstruction);
}

TEST_F(search_tests, require_value_for_every_variable) {
    ASSERT_THROW(Sampler("[N,H,W]", "[N,H,W]", {"N=3:0"}, {"k_1=3:4", "s_1=2"}, {{}}, {{0, 0}}), std::runtime_error);
}

} // namespace kas
