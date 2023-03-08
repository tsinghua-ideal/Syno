#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "KAS/Utils/Algorithm.hpp"
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

TEST(utils_tests, weak_ordered_binary_search) {
    struct Item {
        int key1;
        int key2;
        std::weak_ordering operator<=>(const Item& other) const noexcept {
            return key1 <=> other.key1;
        }
        bool operator==(const Item& other) const noexcept {
            return key1 == other.key1 && key2 == other.key2;
        }
    };
    std::vector<Item> v1 {{1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {3, 0}, {4, 0}, {5, 0}};
    for (std::size_t i = 0; const Item& item: v1) {
        auto it = WeakOrderedBinarySearch(v1, item);
        ASSERT_NE(it, v1.end());
        ASSERT_EQ(it - v1.begin(), i);
        ++i;
    }
    ASSERT_EQ(WeakOrderedBinarySearch(v1, Item {0, 0}), v1.end());
    ASSERT_EQ(WeakOrderedBinarySearch(v1, Item {6, 0}), v1.end());
    std::vector<Item> v2 {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}};
    for (std::size_t i = 0; const Item& item: v2) {
        auto it = WeakOrderedBinarySearch(v2, item);
        ASSERT_NE(it, v2.end());
        ASSERT_EQ(it - v2.begin(), i);
        ++i;
    }
    ASSERT_EQ(WeakOrderedBinarySearch(v2, Item {0, -1}), v2.end());
    ASSERT_EQ(WeakOrderedBinarySearch(v2, Item {0, 6}), v2.end());
}
