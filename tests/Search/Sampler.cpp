#include "Prelude.hpp"


namespace kas {

TEST_F(search_tests, sampler) {
    auto rng = std::mt19937_64(42);

    constexpr std::size_t trials = 50;
    std::size_t successes = 0;
    std::size_t successfulReconstruction = 0;
    std::size_t failedReconstruction = 0;
    for (int i = 0; i < trials; ++i) {
        auto randomLeaves = sampler.randomFinalNodesWithPrefix({}, 32);
        std::erase_if(randomLeaves, [](const auto& pair) { return pair.first.size() < 3; });

        for (auto& [sampledPath, node]: randomLeaves) {
            auto path = sampler.convertTensorViewToPath(node.asFinalStage()->value);
            auto reconstructedNode = sampler.visit(path);
            if (reconstructedNode.has_value()) {
                ASSERT_EQ(
                    node.asFinalStage()->value.printNestedLoopsForAll(ctx),
                    reconstructedNode->asFinalStage()->value.printNestedLoopsForAll(ctx)
                );
                ++successfulReconstruction;
            } else {
                ++failedReconstruction;
            }
            fmt::print("Expanding lattice... ");
            sampler.visit({})->expandToSync(node);
            fmt::print("Done.\n");
            auto finalFlops = node.asFinalStage()->value.getFLOPs(ctx);
            while (!sampledPath.empty()) {
                sampledPath.pop_back();
                auto [_, flops] = sampler.visit(sampledPath)->getShapeDistance();
                if (flops > finalFlops) {
                    fmt::print("Oh, no! FLOPs-based pruning fails!");
                    fmt::print("Final DFG:\n{}", GraphvizDFGGen::Print(node.asFinalStage()->value.getSubgraphs(), ctx));
                    fmt::print("Intermediate:\n{}", GraphvizGen::Print(sampler.visit(sampledPath).value().asNonFinalStage()->getInterface().getRaw(), ctx));
                }
                ASSERT_LE(flops, finalFlops);
            }
        }
        if (randomLeaves.empty()) {
            fmt::print("Trial {} failed.\n", i);
            continue;
        } else {
            fmt::print("Trial {} succeeded.\n", i);
        }
        auto randomLeafIndex = std::uniform_int_distribution<std::size_t>(0, randomLeaves.size() - 1)(rng);
        auto [_, node] = randomLeaves[randomLeafIndex];
        ++successes;
        auto& tensorView = node.asFinalStage()->value;

        auto r = tensorView.getUnderlyingTensors() | std::ranges::views::transform([&](const auto& tensor) { return tensor.shapeToString(ctx); });
        std::cout << fmt::format("Input Shape: {}", fmt::join(r, ", ")) << std::endl;
        std::cout << tensorView.printNestedLoopsForAll(ctx);

        PyTorchGen(ctx, tensorView).generateSingle("./search_pt/trial_" + std::to_string(i) + ".py", "trial_" + std::to_string(i), tensorView, dict);
        GraphvizGen(tensorView, ctx).generate("./search_viz/trial_" + std::to_string(i) + ".dot", "trial_" + std::to_string(i));

#ifdef KAS_USE_HALIDE
        if (doRealization) {
            auto cgOpt = CodeGenOptions();
            cgOpt.scheduler = CodeGenOptions::AutoScheduler::Anderson2021;
            cgOpt.useGPU = true;
            HalideGen gen(ctx, tensorView, cgOpt);
            auto name = "search_codegen_test_" + std::to_string(i);
            gen.performTrial<false>(dict, name, true, false, []{});
        }
#endif
    }
    StatisticsCollector::PrintSummary(std::cout);
    fmt::print("{}", sampler.statsToString());
    fmt::print("Success rate: {:.2f} ({} / {})\n", static_cast<float>(successes) / trials, successes, trials);
    fmt::print("Reconstruction: successful = {}, failed = {}\n", successfulReconstruction, failedReconstruction);
    ASSERT_GT(successfulReconstruction, failedReconstruction);
}

TEST_F(search_tests, require_value_for_every_variable) {
    ASSERT_THROW(Sampler("[N,H,W]", "[N,H,W]", {"N=3:0"}, {"k_1=3:4", "s_1=2"}, {{}}, {{0, 0}}), std::runtime_error);
}

} // namespace kas
