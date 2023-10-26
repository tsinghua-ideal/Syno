#include "Prelude.hpp"


namespace kas {

TEST_F(search_tests, forward) {
    Forward::Factory factory { ctx };
    auto [sizeN, sizeH, sizeW, sizeK] = factory.getSizes("N", "H", "W", "k_1");

    auto [dimN, dimH, dimW] = factory.makeDimsOfSizes(sizeN, sizeH, sizeW);
    // [N, H, W], the input.

    auto [dimK1, dimK2] = factory.makeDimsOfSizes(sizeK, sizeK);
    // [K, K], the filter.

    // The input tensors are blended into [N, H, W, K, K].

    auto [dimH_over_K, dimH_dot_K] = Forward::UnfoldOp::Create(dimH, sizeK);
    auto [dimW_over_K, dimW_dot_K] = Forward::UnfoldOp::Create(dimW, sizeK);
    // [N, H, K, W, K, K, K], where H and W are unfolded.

    auto dimK1_shared = Forward::ShareOp::Create(dimH_dot_K, dimK1);
    auto dimK2_shared = Forward::ShareOp::Create(dimW_dot_K, dimK2);
    // [N, H, W, K, K], where K1, and K2 are shared.

    dimN.output(0);
    dimH_over_K.output(1);
    dimW_over_K.output(2);
    dimK2_shared.reduce(Reduce::ReduceType::Sum);
    dimK1_shared.reduce(Reduce::ReduceType::Sum);
    // [N, H, W], the output.

    auto input = Topmost({ dimN, dimH, dimW }, {});
    auto weight = Topmost({ dimK1, dimK2 }, {});
    std::vector<Topmost> tensors { input, weight };
    sampler.sortAllExpansionsAndWeightDimensions(tensors);
    auto tensorView = TensorView(tensors, TensorExpression::ProductOfTensors(tensors.size()), ctx);
    auto obtainedTensors = ranges::to<std::vector<Topmost>>(tensorView.getUnderlyingTensors() | std::views::transform(&PureTensor::getContent));
    sampler.sortAllExpansionsAndWeightDimensions(obtainedTensors);
    ASSERT_EQ(tensors.size(), obtainedTensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        ASSERT_EQ(tensors[i].description(ctx), obtainedTensors[i].description(ctx));
    }
    sampler.removeFixedDimensions(tensors);
    auto path = Sampler::ConvertSearchableTensorsToPath(tensors);
    fmt::print("A possible path is:\n");
    for (auto&& next: path) {
        fmt::print("{}\n", next.toString());
    }
    auto node = sampler.visit({}).value();
    for (auto next: path) {
        fmt::print("Trying {}...\n", next.toString());
        fmt::print("The children are:\n");
        for (auto handles = node.getChildrenHandles(); auto handle: handles) {
            fmt::print("  {}\n", node.getChildDescription(handle).value());
        }
        fmt::print("Getting child...\n");
        auto old = node;
        node = node.getChild(next).value();
        fmt::print("Got child. BTW, it is {}.\n", old.getChildDescription(next).value());
    }
    ASSERT_EQ(
        node.asFinalStage()->value.printNestedLoopsForAll(ctx),
        tensorView.printNestedLoopsForAll(ctx)
    );

    auto graph = tensorView.buildGraph();
    const Color& colorN = graph.colorOf(dimN), & colorK1 = graph.colorOf(dimK1), & colorK2 = graph.colorOf(dimK2), & colorH = graph.colorOf(dimH), & colorW = graph.colorOf(dimW);
    ASSERT_EQ(colorN.getHeight(), 0);
    ASSERT_EQ(colorK1.getHeight(), 2);
    ASSERT_EQ(colorK2.getHeight(), 2);
    ASSERT_EQ(colorH.getHeight(), 3);
    ASSERT_EQ(colorW.getHeight(), 3);
}

} // namespace kas
