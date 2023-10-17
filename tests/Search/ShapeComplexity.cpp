#include "Prelude.hpp"


namespace kas {

TEST_F(search_tests, shape_complexity) {

    auto [N, H, W, k_1, s_1] = ctx.getSizes("N", "H", "W", "k_1", "s_1");

    Shape desired = std::vector<Size> {N, H, W};
    ShapeComplexity::DistanceOptions options = {
        .ctx = ctx,
        .remainingMerges = 0,
        .remainingSplits = 0,
        .remainingUnfoldsAndExpands = 0,
        .overflow = 1,
    };

    ASSERT_EQ(ShapeComplexity::Compute(desired, {N, H, W}, options), 0);

    ASSERT_EQ(ShapeComplexity::Compute(desired, {N, H * W}, options), ShapeComplexity::Infinity);
    options.remainingMerges = 1;
    ASSERT_EQ(ShapeComplexity::Compute(desired, {N, H * W}, options), 1);

    ASSERT_EQ(ShapeComplexity::Compute(desired, {N, H, W, k_1}, options), ShapeComplexity::Infinity);
    options.remainingUnfoldsAndExpands = 1;
    ASSERT_EQ(ShapeComplexity::Compute(desired, {N, H, W, k_1}, options), 1);

    ASSERT_EQ(ShapeComplexity::Compute(desired, {N, H, W / k_1, k_1}, options), ShapeComplexity::Infinity);
    options.remainingSplits = 1;
    ASSERT_EQ(ShapeComplexity::Compute(desired, {N, H, W / k_1, k_1}, options), 1);
}

} // namespace kas
