#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, reshape_canonicalizer) {
    // Basically we use isAdjacentTo() to check if we are allowed to use Split to combine the two dims.
    auto canonicalizer = ReshapeCanonicalizer();
    auto merge = MergeOp(&itH, sizeC);
    {
        // This conteracts the Merge.
        auto l = canonicalizer[merge.getInputL()];
        auto r = canonicalizer[merge.getInputR()];
        ASSERT_TRUE(l.isAdjacentTo(r));
    }
    auto share_w1_r = ShareOp(merge.getInputR(), 1);
    {
        // This does not conteract the Merge.
        auto l = canonicalizer[merge.getInputL()];
        auto r = canonicalizer[share_w1_r.getInputL()];
        ASSERT_FALSE(l.isAdjacentTo(r));
    }
    auto share_w1_l = ShareOp(merge.getInputL(), 1);
    {
        // This conteracts the Merge if we alter the layout of the weight, which is arbitrary.
        auto l = canonicalizer[share_w1_l.getInputL()];
        auto r = canonicalizer[share_w1_r.getInputL()];
        ASSERT_TRUE(l.isAdjacentTo(r));
    }
    auto share_w2 = ShareOp(share_w1_r.getInputL(), 2);
    {
        // This conteracts the Merge, if we swap the two weights.
        auto l = canonicalizer[share_w1_l.getInputL()];
        auto r = canonicalizer[share_w2.getInputL()];
        ASSERT_TRUE(l.isAdjacentTo(r));
    }
}

} // namespace kas
