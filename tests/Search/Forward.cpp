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
    dimK2_shared.reduce(0, MapReduce::MapType::Identity, MapReduce::ReduceType::Mean);
    dimK1_shared.reduce(1, MapReduce::MapType::Identity, MapReduce::ReduceType::Mean);
    // [N, H, W], the output.

    std::vector<Dimension> input { dimN, dimH, dimW }, weight { dimK1, dimK2 };
    std::vector<std::vector<Dimension>> tensors { input, weight };
    Sampler::ConvertTensorViewToSearchableOrder(tensors);
    auto tensorView = TensorView(tensors, TensorExpression::ProductOfTensors(tensors.size()));
    auto path = sampler.convertTensorsToPath(tensors);
    fmt::print("A possible path is:\n");
    for (auto&& next: path) {
        fmt::print("{}\n", next.toString());
    }
    auto node = *sampler.visit({});
    for (auto next: path) {
        fmt::print("Trying {}...\n", next.toString());
        fmt::print("The children are:\n");
        for (auto handles = node.getChildrenHandles(); auto handle: handles) {
            fmt::print("  {}\n", *node.getChildDescription(handle));
        }
        fmt::print("Getting child...\n");
        auto old = node;
        node = *node.getChild(next);
        fmt::print("Got child. BTW, it is {}.\n", *old.getChildDescription(next));
    }
    ASSERT_EQ(
        node.asFinal()->printNestedLoopsForAll(ctx),
        tensorView.printNestedLoopsForAll(ctx)
    );
}

} // namespace kas
