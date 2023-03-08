#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "KAS/Utils/Algorithm.hpp"
#include "KAS/Utils/Vector.hpp"


using namespace kas;

struct WeakOrderedItem {
    int key1;
    int key2;
    std::weak_ordering operator<=>(const WeakOrderedItem& other) const noexcept {
        return key1 <=> other.key1;
    }
    bool operator==(const WeakOrderedItem& other) const noexcept {
        return key1 == other.key1 && key2 == other.key2;
    }
};

TEST(utils_tests, weak_ordered_binary_search) {
    std::vector<WeakOrderedItem> v1 {{1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {3, 0}, {4, 0}, {5, 0}};
    for (std::size_t i = 0; const WeakOrderedItem& item: v1) {
        auto it = WeakOrderedBinarySearch(v1, item);
        ASSERT_NE(it, v1.end());
        ASSERT_EQ(it - v1.begin(), i);
        ++i;
    }
    ASSERT_EQ(WeakOrderedBinarySearch(v1, WeakOrderedItem {0, 0}), v1.end());
    ASSERT_EQ(WeakOrderedBinarySearch(v1, WeakOrderedItem {6, 0}), v1.end());
    std::vector<WeakOrderedItem> v2 {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}};
    for (std::size_t i = 0; const WeakOrderedItem& item: v2) {
        auto it = WeakOrderedBinarySearch(v2, item);
        ASSERT_NE(it, v2.end());
        ASSERT_EQ(it - v2.begin(), i);
        ++i;
    }
    ASSERT_EQ(WeakOrderedBinarySearch(v2, WeakOrderedItem {0, -1}), v2.end());
    ASSERT_EQ(WeakOrderedBinarySearch(v2, WeakOrderedItem {0, 6}), v2.end());
}

TEST(utils_tests, weak_ordered_substitute_vector) {
    struct FullItem {
        WeakOrderedItem key;
        int metadata;
        struct Project {
            const WeakOrderedItem& operator()(const FullItem& item) const noexcept {
                return item.key;
            }
        };
    };
    std::vector<FullItem> v1 {{{1, 1}, 10}, {{1, 2}, 10}, {{1, 3}, 10}, {{2, 1}, 10}, {{3, 1}, 10}};
    constexpr int update = 20;
    auto consistent = [](const std::vector<FullItem>& v, bool expectUpdate) {
        bool hasUpdatedMetadata = false;
        if (v[0].metadata == update) {
            hasUpdatedMetadata = true;
        }
        for (std::size_t i = 1; i < v.size(); ++i) {
            if (v[i - 1].key > v[i].key) {
                fmt::print("v[{}] = {{{}, {}}}, v[{}] = {{{}, {}}}\n", i - 1, v[i - 1].key.key1, v[i - 1].key.key2, i, v[i].key.key1, v[i].key.key2);
                return false;
            }
            if (v[i].metadata == update) {
                hasUpdatedMetadata = true;
            }
        }
        if (expectUpdate != hasUpdatedMetadata) {
            fmt::print("hasUpdatedMetadata = {}\n", hasUpdatedMetadata);
            return false;
        }
        return true;
    };
    for (std::size_t i = 0; i < v1.size(); ++i) {
        fmt::print("Verification i = {} (key = {{{}, {}}})\n", i, v1[i].key.key1, v1[i].key.key2);
        auto res = v1;
        WeakOrderedSubstituteVector1To1IfAny(res, v1[i].key, FullItem { {static_cast<int>(i), static_cast<int>(i)}, update }, {}, FullItem::Project{});
        ASSERT_TRUE(consistent(res, true));
        res = v1;
        WeakOrderedSubstituteVector1To1IfAny(res, WeakOrderedItem { v1[i].key.key1, 100 }, FullItem { {static_cast<int>(i), static_cast<int>(i)}, update }, {}, FullItem::Project{});
        ASSERT_TRUE(consistent(res, false));
    }
}
