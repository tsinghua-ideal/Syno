#include "Prelude.hpp"


namespace kas {

TEST_F(search_tests, shape_complexity) {

    auto [N, H, W, k_1, s_1] = ctx.getSizes("N", "H", "W", "k_1", "s_1");

    Shape desired = std::vector<Size> {N, H, W};
    FinalizeOp::DistanceOptions options = {
        .ctx = ctx,
        .remainingMerges = 0,
        .remainingSplits = 0,
        .remainingUnfolds = 0,
    };
    constexpr std::size_t Infinity = std::numeric_limits<std::size_t>::max();

    ASSERT_EQ(FinalizeOp::ShapeComplexity(desired, {N, H, W}, options), 0);

    ASSERT_EQ(FinalizeOp::ShapeComplexity(desired, {N, H * W}, options), Infinity);
    options.remainingMerges = 1;
    ASSERT_EQ(FinalizeOp::ShapeComplexity(desired, {N, H * W}, options), 1);

    ASSERT_EQ(FinalizeOp::ShapeComplexity(desired, {N, H, W, k_1}, options), Infinity);
    options.remainingUnfolds = 1;
    ASSERT_EQ(FinalizeOp::ShapeComplexity(desired, {N, H, W, k_1}, options), 1);

    ASSERT_EQ(FinalizeOp::ShapeComplexity(desired, {N, H, W / k_1, k_1}, options), Infinity);
    options.remainingSplits = 1;
    ASSERT_EQ(FinalizeOp::ShapeComplexity(desired, {N, H, W / k_1, k_1}, options), 1);
}

} // namespace kas
