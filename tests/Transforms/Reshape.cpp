#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, reshape_canonicalizer) {
    auto canonicalizer = ReshapeCanonicalizer();
    auto merge = MergeOp(&itH, sizeC);
    {
        auto l = canonicalizer[merge.getInputL()];
        auto r = canonicalizer[merge.getInputR()];
        ASSERT_TRUE(l.isAdjacentTo(r));
    }
    auto share1 = ShareOp(merge.getInputR(), 1);
    {
        auto l = canonicalizer[merge.getInputL()];
        auto r = canonicalizer[share1.getInputL()];
        ASSERT_FALSE(l.isAdjacentTo(r));
    }
    auto share2 = ShareOp(merge.getInputL(), 1);
    {
        auto l = canonicalizer[share2.getInputL()];
        auto r = canonicalizer[share1.getInputL()];
        ASSERT_TRUE(l.isAdjacentTo(r));
    }
}

} // namespace kas
