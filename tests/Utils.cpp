#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

#include "KAS/Utils/Vector.hpp"


using namespace kas;

TEST(utils_tests, replace_vector) {
    std::vector<int> vec { 0, 1, 2, 3, 4, 5 };
    std::vector<int> drops { 2, 5 };
    std::vector<std::pair<int, int>> adds { { 1, 10 }, { 2, 20 }, { 4, 40 } };
    auto newVec = ReplaceVector<int>(vec, drops, adds);
    std::vector<int> expected { 0, 10, 20, 1, 40, 3, 4 };
    ASSERT_EQ(newVec, expected);
}
