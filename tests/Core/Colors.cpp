#include "gtest/gtest.h"

#include "KAS/Core/Colors.hpp"


using namespace kas;

TEST(core_colors_tests, remove_tags) {
    using Tag = Color::Tag;

    Tag red = reinterpret_cast<Tag>(0), green = reinterpret_cast<Tag>(1), blue = reinterpret_cast<Tag>(2), yellow = reinterpret_cast<Tag>(3);

    // Test removing a single tag
    std::vector<Tag> tags1 = {red, green, blue};
    std::vector<Tag> toRemove1 = {green};
    std::size_t removed1 = Color::RemoveTags(tags1, toRemove1);
    EXPECT_EQ(removed1, 1);
    EXPECT_EQ(tags1.size(), 2);
    EXPECT_EQ(tags1[0], red);
    EXPECT_EQ(tags1[1], blue);

    // Test removing multiple tags
    std::vector<Tag> tags2 = {red, green, blue, yellow};
    std::vector<Tag> toRemove2 = {green, yellow};
    std::size_t removed2 = Color::RemoveTags(tags2, toRemove2);
    EXPECT_EQ(removed2, 2);
    EXPECT_EQ(tags2.size(), 2);
    EXPECT_EQ(tags2[0], red);
    EXPECT_EQ(tags2[1], blue);

    // Test removing no tags
    std::vector<Tag> tags3 = {red, green, blue};
    std::vector<Tag> toRemove3 = {yellow};
    std::size_t removed3 = Color::RemoveTags(tags3, toRemove3);
    EXPECT_EQ(removed3, 0);
    EXPECT_EQ(tags3.size(), 3);
    EXPECT_EQ(tags3[0], red);
    EXPECT_EQ(tags3[1], green);
    EXPECT_EQ(tags3[2], blue);

    // Test removing all tags
    std::vector<Tag> tags4 = {red, green, blue};
    std::vector<Tag> toRemove4 = {red, green, blue};
    std::size_t removed4 = Color::RemoveTags(tags4, toRemove4);
    EXPECT_EQ(removed4, 3);
    EXPECT_EQ(tags4.size(), 0);
}
