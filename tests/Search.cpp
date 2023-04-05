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
        .dimLowerBound = 4,
        .dimUpperBound = 8,
        .maximumTensors = 2,
    };
    Sampler sampler("[N,C,H,W]", "[N,C,H,W]", {}, {"k_1=3", "s_1=2", "k_2=5", "s_2=4"}, options);
    auto& ctx = sampler.getBindingContext();
    ASSERT_EQ(ctx.getPrimaryCount(), 4);
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
/*
        constexpr int dimH = 4, dimW = 4, dimN = 4, dimC = 4, dimK1 = 3, dimS1 = 2, dimK2 = 3, dimS2 = 2;
        HalideGen gen(ctx, tensorView, HalideGen::Options());
        auto name = "search_codegen_test_" + std::to_string(i);
        std::map<std::string, std::size_t> dict { { "H", dimH }, { "W", dimW }, { "N", dimN }, { "C", dimC }, { "k_1", dimK1 }, { "s_1", dimS1 }, { "k_2", dimK2 }, { "s_2", dimS2 } };
        auto consts = ctx.realizeConsts(dict);
        auto access = gen.evaluateAccess(consts);
        auto [inputs, func] = gen.createFunc(consts, access, name);
        ASSERT_EQ(inputs.size(), 2);
        auto& input = inputs[0], & weight = inputs[1];
        auto getInputsShape = [&](int i) -> std::vector<std::size_t> {
            return tensorView.getUnderlyingTensors()[i].getShape().eval<std::size_t>(consts);
        };
        auto inputShape = getInputsShape(0), weightShape = getInputsShape(1);
        auto expectedInputShape = std::vector<std::size_t> {dimH, dimW};
        ASSERT_EQ(inputShape, expectedInputShape);
        // Revert the order of shape due to column-major layout.
        auto inputBuffer = Halide::Buffer<float, 2>(std::vector<int>(inputShape.rbegin(), inputShape.rend()));
        inputBuffer.for_each_value([=](float& v) {
            static int cnt = 0;
            v = cnt++;
        });
        input.set(inputBuffer);
        auto weightBuffer = Halide::Buffer<float>(std::vector<int>(weightShape.rbegin(), weightShape.rend()));
        weightBuffer.for_each_value([=](float& v) {
            static int cnt = 0;
            v = cnt++;
        });
        weight.set(weightBuffer);
        func.compute_root();
        func.trace_stores();
        auto outputShape = std::vector<int> {dimN, dimC, dimH, dimW};
        Halide::Buffer<float, 4> outputBuffer = func.realize(std::vector<int>(outputShape.rbegin(), outputShape.rend()), Halide::get_host_target());
*/
    }
    StatisticsCollector::PrintSummary(std::cout);
    fmt::print("Success rate: {:.2f} ({} / {})\n", static_cast<float>(successes) / trials, successes, trials);
}

} // namespace kas
