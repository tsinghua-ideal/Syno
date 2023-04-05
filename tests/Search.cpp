#include <map>
#include <memory>
#include <ranges>
#include <string>
#include <vector>

#include <fmt/ranges.h>
#include <gtest/gtest.h>

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
    SampleOptions options {
        .seed = 42,
        .depth = 4,
        .dimLowerBound = 2,
        .dimUpperBound = 6,
        .maximumTensors = 2,
    };
    Sampler sampler("[H,W]", "[H,W]", {}, {"k_1=3", "s_1=2", "k_2=5", "s_2=4"}, options);
    auto& ctx = sampler.getBindingContext();
    BindingContext::PublicCtx = &ctx; // For debugging.
    ASSERT_EQ(ctx.getPrimaryCount(), 2);
    ASSERT_EQ(ctx.getCoefficientCount(), 4);

    constexpr std::size_t trials = 100;
    std::size_t successes = 0;
    for (int i = 0; i < trials; ++i) {
        auto path = sampler.randomPathWithPrefix({});
        if (!sampler.isFinal(path)) {
            fmt::print("Trial {} failed.\n", i);
            continue;
        }
        ++successes;
        auto& tensorView = *sampler.realize(path);

        auto r = tensorView.getUnderlyingTensors() | std::ranges::views::transform([&](const auto& tensor) { return tensor.shapeToString(ctx); });
        std::cout << fmt::format("Input Shape: {}", fmt::join(r, ", ")) << std::endl;
        std::cout << tensorView.printNestedLoops(ctx, AbstractAccess::Output);

        constexpr int dimH = 64, dimW = 64, dimK1 = 3, dimS1 = 2, dimK2 = 5, dimS2 = 4;
        HalideGen gen(ctx, tensorView, HalideGen::Options());
        auto name = "search_codegen_test_" + std::to_string(i);
        std::map<std::string, std::size_t> dict { { "H", dimH }, { "W", dimW }, { "k_1", dimK1 }, { "s_1", dimS1 }, { "k_2", dimK2 }, { "s_2", dimS2 } };
        gen.performTrial<false>(dict, name, false,
            [](){}, [](){}, [](){}
        );
    }
    StatisticsCollector::PrintSummary(std::cout);
    fmt::print("Success rate: {:.2f} ({} / {})\n", static_cast<float>(successes) / trials, successes, trials);
}

} // namespace kas
