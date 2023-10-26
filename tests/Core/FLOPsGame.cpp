#include "gtest/gtest.h"

#include "KAS/Core/FLOPsGame.hpp"


using namespace kas;

TEST(core_flops_game_tests, independent) {
    constexpr int S = 2, K_1 = 3, K_2 = 5;
    BindingContext ctx = BindingContext({}, {"s=2", "k_1=3", "k_2=5"});
    auto [s, k_1, k_2] = ctx.getSizes("s", "k_1", "k_2");
    auto game = FLOPsGame {
        .ctx = ctx,
        .inputSize = s * k_2,
        .increase = { s, k_1 },
        .decrease = { k_2, s, s * k_1 },
        .dependencies = {
            { false, false, },
            { true, false, },
            { false, true, },
        },
    };
    // Best scheme: first k_2, then s*k_1, then s.
    // s * k_2
    // reduce k_2, FLOPs = s * k_2
    // s
    // expand k_1, reduce s * k_1, FLOPs = s * k_1
    // 1
    // expand s, reduce s, FLOPs = s
    // 1
    // Total FLOPs = s * k_2 + s * k_1 + s = 18
    ASSERT_EQ(game.FLOPs(), S * K_2 + S * K_1 + S);
}

TEST(core_flops_game_tests, dependent) {
    constexpr int K_1 = 3, K_2 = 5;
    BindingContext ctx = BindingContext({}, {"s=2", "k_1=3", "k_2=5"});
    auto [k_1, k_2] = ctx.getSizes("k_1", "k_2");
    auto game = FLOPsGame {
        .ctx = ctx,
        .inputSize = k_1,
        .increase = { k_2 },
        .decrease = { k_1, k_2 },
        .dependencies = {
            { true, },
            { true, },
        },
    };
    // Best scheme: do them together.
    ASSERT_EQ(game.FLOPs(), K_1 * K_2);
}
