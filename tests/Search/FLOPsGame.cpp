#include <gtest/gtest.h>

#include "KAS/Search/FLOPsGame.hpp"
#include "KAS/Transforms/Transforms.hpp"


namespace kas {

using namespace std::string_literals;

class search_flops_game_tests: public ::testing::Test {
protected:
    static constexpr int N = 3, H = 16, W = 16, C_in = 3, C_out = 8, K1 = 3, K2 = 5, S1 = 2;
    BindingContext ctx = BindingContext({
        "N=3"s,"H=16"s, "W=16"s, "C_out=8"s, "C_in=3"s,
    }, {
        "k_1=3"s, "s_1=2"s,
    });
    Iterator i_C_out { 1, ctx.getSize("C_out") }, i_H { 2, ctx.getSize("H") }, i_W { 3, ctx.getSize("W") };
    ReduceOp r_C_in { ctx.getSize("C_in"), Reduce::ReduceType::Sum }, r_k_1 { ctx.getSize("k_1"), Reduce::ReduceType::Sum };
    search_flops_game_tests() {
        ctx.debug(); // For debugging.
    }
};

TEST_F(search_flops_game_tests, outer_product) {
    auto interface = GraphHandle({&i_C_out, &i_H, &i_W, r_C_in.getInput(0)}, {});
    auto graph = interface.buildGraph();
    auto extendedGame = ExtendedFLOPsGame(ctx, ctx.getSize("N*C_in*H*W"), graph);
    auto weight = std::vector<Dimension>{&i_C_out};
    auto game = extendedGame.getGameWithWeights({weight});
    auto flops = game.FLOPs();
    auto expectedFLOPs =
        // Early reduction. Yields [N, H, W].
        N * C_in * H * W;
        // Outer product is ignored. Because it does not contribute to rfactor.
        // N * C_out * H * W;
    ASSERT_EQ(flops, expectedFLOPs);
}

TEST_F(search_flops_game_tests, dilated_conv) {
    auto strideOp = StrideOp(r_k_1.getInput(0), ctx.getSize("k_1"));
    auto unfoldOp = UnfoldOp(&i_H, strideOp.getInput());
    auto interface = GraphHandle({unfoldOp.getInput()}, {});
    auto graph = interface.buildGraph();
    auto extendedGame = ExtendedFLOPsGame(ctx, ctx.getSize("H"), graph);
    auto game = extendedGame.getGameWithWeights({});
    auto flops = game.FLOPs();
    auto expectedFLOPs = H * K1;
    ASSERT_EQ(flops, expectedFLOPs);
}

} // namespace kas
