#include <gtest/gtest.h>

#include "KAS/Search/Finalize.hpp"


namespace kas {

class search_finalize_tests: public ::testing::Test {
protected:
    
};

TEST_F(search_finalize_tests, empty_interface) {
    std::vector<ColoredDimension> remaining;
    auto gen = FinalizeOp::AssignToWeights(remaining, 1);
    auto it = gen.begin();
    ASSERT_TRUE(it != gen.end());
    ASSERT_EQ((*it).size(), 0);
    ASSERT_EQ(++it, gen.end());
}

} // namespace kas
