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
#include "KAS/Transforms.hpp"


namespace kas {

TEST(search_tests, sampler) {
    SampleOptions options;
    options.seed = 42;
    options.depth = 4;
    options.dimLowerBound = 4;
    options.dimUpperBound = 8;
    Sampler sampler("[H,W]", "[N,C,H,W]", {}, {"k_1", "s_1", "k_2", "s_2"}, options);
    auto& ctx = sampler.getBindingContext();
    ASSERT_EQ(ctx.getPrimaryCount(), 4);
    ASSERT_EQ(ctx.getCoefficientCount(), 4);
    for (int i = 0; i < 10; ++i) {
        auto path = sampler.randomPathWithPrefix({});
        ASSERT_EQ(sampler.isFinal(path), true); // Well, this could fail, actually.
        auto& tensorView = *sampler.realize(path);

        auto r = tensorView.getUnderlyingTensors() | std::ranges::views::transform([&](const auto& tensor) { return tensor.shapeToString(ctx); });
        std::cout << fmt::format("Input Shape: {}", fmt::join(r, ", ")) << std::endl;
        std::cout << tensorView.printNestedLoops(ctx);

        constexpr int dimH = 4, dimW = 4, dimN = 4, dimC = 4, dimK1 = 3, dimS1 = 2, dimK2 = 3, dimS2 = 2;
        HalideGen gen(ctx, tensorView);
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
    }
}

} // namespace kas
