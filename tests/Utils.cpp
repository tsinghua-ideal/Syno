#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

#include "KAS/Utils/Vector.hpp"


using namespace kas;

TEST(utils_tests, replace_vector) {
    std::vector<std::size_t> vec { 0, 1, 2, 3, 4, 5 };
    std::vector<std::size_t> drops { 2, 5 };
    std::vector<std::pair<std::size_t, std::size_t>> adds { { 1, 10 }, { 2, 20 }, { 4, 40 } };
    auto newVec = ReplaceVector<std::size_t>(vec, drops, adds);
    std::vector<std::size_t> expected { 0, 10, 20, 1, 40, 3, 4 };
    ASSERT_EQ(newVec, expected);
}
