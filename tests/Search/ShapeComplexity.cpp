#include "Prelude.hpp"


namespace kas {

TEST_F(search_tests, shape_complexity) {

    auto [N, H, W, k_1, s_1] = ctx.getSizes("N", "H", "W", "k_1", "s_1");

    auto desired = std::vector<DesiredSize> {{N}, {H}, {W}};
    ShapeComplexity::DistanceOptions options = {
        .ctx = ctx,
        .requiresOnlyOddNumelIncrease = false,
        .remainingMerges = 0,
        .remainingSplits = 0,
        .remainingUnfoldsAndExpands = 0,
        .overflow = 1,
    };

    ASSERT_EQ(ShapeComplexity::Compute(desired, {{N, 0}, {H, 1}, {W, 1}}, options), 0);

    ASSERT_EQ(ShapeComplexity::Compute(desired, {{N, 1}, {H * W, 1}}, options), ShapeComplexity::Infinity);
    options.remainingMerges = 1;
    ASSERT_EQ(ShapeComplexity::Compute(desired, {{N, 0}, {H * W, 1}}, options), 1);

    ASSERT_EQ(ShapeComplexity::Compute(desired, {{N, 1}, {H, 1}, {W, 1}, {k_1, 1}}, options), ShapeComplexity::Infinity);
    options.remainingUnfoldsAndExpands = 1;
    ASSERT_EQ(ShapeComplexity::Compute(desired, {{N, 0}, {H, 0}, {W, 1}, {k_1, 1}}, options), 1);

    ASSERT_EQ(ShapeComplexity::Compute(desired, {{N, 1}, {H, 1}, {W / k_1, 1}, {k_1, 1}}, options), ShapeComplexity::Infinity);
    options.remainingSplits = 1;
    ASSERT_EQ(ShapeComplexity::Compute(desired, {{N, 0}, {H, 0}, {W / k_1, 1}, {k_1, 1}}, options), 1);
}

} // namespace kas
