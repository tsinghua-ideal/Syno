#include "Prelude.hpp"


namespace kas {

TEST_F(search_tests, forward) {
    Forward::Factory factory { ctx };
    auto [sizeN, sizeH, sizeW, sizeK, sizeC_in, sizeC_out] = factory.getSizes("N", "H", "W", "k_1", "C_in", "C_out");

    auto [dimN, dimC_in_in, dimH, dimW] = factory.makeDimsOfSizes(sizeN, sizeC_in, sizeH, sizeW);
    // [N, C_in, H, W], the input.

    auto [dimC_out_w, dimC_in_w, dimK1, dimK2] = factory.makeDimsOfSizes(sizeC_out, sizeC_in, sizeK, sizeK);
    // [C_out, C_in, K, K], the filter.

    // The input tensors are blended into [N, C_in, H, W, C_out, C_in, K, K].

    auto [dimH_over_K, dimH_dot_K] = Forward::UnfoldOp::Create(dimH, sizeK);
    auto [dimW_over_K, dimW_dot_K] = Forward::UnfoldOp::Create(dimW, sizeK);
    // [N, C_in, H, K, W, K, C_out, C_in, K, K], where H and W are unfolded.

    auto dimK1_shared = Forward::ShareOp::Create(dimH_dot_K, dimK1);
    auto dimK2_shared = Forward::ShareOp::Create(dimW_dot_K, dimK2);
    auto dimC_in_shared = Forward::ShareOp::Create(dimC_in_in, dimC_in_w);
    // [N, H, W, C_out, C_in, K, K], where K1, K2 and C_in are shared.

    auto dimC_out_expand = Forward::ExpandOp::Create(factory, sizeC_out);
    auto dimC_out = Forward::ShareOp::Create(dimC_out_expand, dimC_out_w);
    // [N, C_out, H, W, C_in, K, K], where C_out is merged.

    dimN.output(0);
    dimC_out.output(1);
    dimH_over_K.output(2);
    dimW_over_K.output(3);
    dimC_in_shared.reduce(Reduce::ReduceType::Sum);
    dimK1_shared.reduce(Reduce::ReduceType::Sum);
    dimK2_shared.reduce(Reduce::ReduceType::Sum);
    // [N, C_out, H, W], the output.

    factory.inputs({
        {dimN, dimC_in_in, dimH, dimW, dimC_out_expand},
        {dimC_out_w, dimC_in_w, dimK1, dimK2},
    });
    std::vector<Topmost> tensors = factory.getInputs();
    sampler.sortAllExpansionsAndWeightDimensions(tensors);
    auto tensorView = TensorView(tensors, TensorExpression::ProductOfTensors(tensors.size()), ctx);
    auto obtainedTensors = ranges::to<std::vector<Topmost>>(tensorView.getUnderlyingTensors() | std::views::transform(&PureTensor::getContent));
    sampler.sortAllExpansionsAndWeightDimensions(obtainedTensors);
    ASSERT_EQ(tensors.size(), obtainedTensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        ASSERT_EQ(tensors[i].description(ctx), obtainedTensors[i].description(ctx));
    }
    sampler.removeFixedDimensions(tensors);
    auto path = Sampler::ConvertSearchableTensorsToPath(tensors, sampler.getOpStore());
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
