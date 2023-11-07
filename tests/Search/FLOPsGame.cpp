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
    ReduceOp r_C_in { ctx.getSize("C_in"), Reduce::ReduceType::Sum }, r_k_1 { ctx.getSize("k_1"), Reduce::ReduceType::Sum }, r_s_1 { ctx.getSize("s_1"), Reduce::ReduceType::Sum };
    search_flops_game_tests() {
        ctx.debug(); // For debugging.
    }
    std::size_t getDeducedFLOPs(const GraphHandle& interface, const Size& inputSize, const std::vector<std::vector<Dimension>>& weights) const {
        auto graph = interface.buildGraph();
        auto extendedGame = ExtendedFLOPsGame(ctx, inputSize, graph);
        auto game = extendedGame.getGameWithWeights(weights);
        return game.FLOPs();
    }
};

TEST_F(search_flops_game_tests, outer_product) {
    auto interface = GraphHandle({&i_C_out, &i_H, &i_W, r_C_in.getInput(0)}, {});
    auto flops = getDeducedFLOPs(interface, ctx.getSize("N*C_in*H*W"), {{&i_C_out}});
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
    auto flops = getDeducedFLOPs(interface, ctx.getSize("H"), {});
    auto expectedFLOPs = H * K1;
    ASSERT_EQ(flops, expectedFLOPs);
}

TEST_F(search_flops_game_tests, conv) {
    // TODO! -1 is ugly.
    auto shareOp_k = ShareOp(r_k_1.getInput(0), -1);
    auto unfoldOp = UnfoldOp(&i_H, shareOp_k.getInputL());
    auto shareOp_s_bottom = ShareOp(r_s_1.getInput(0), -1);
    auto shareOp_s_top = ShareOp(shareOp_s_bottom.getInputL(), -1);
    auto expandOp = ExpandOp(shareOp_s_top.getInputL());
    auto interface = GraphHandle({unfoldOp.getInput(), shareOp_k.getInputR(), shareOp_s_top.getInputR(), shareOp_s_bottom.getInputR(), &i_C_out}, {&expandOp});
    auto weightTop = std::vector<Dimension> {shareOp_k.getInputR(), shareOp_s_top.getInputR()};
    auto weightBottom = std::vector<Dimension> {shareOp_s_bottom.getInputR(), &i_C_out};
    auto flops = getDeducedFLOPs(interface, ctx.getSize("H"), {weightTop, weightBottom});
    auto expectedFLOPs = H * K1 * S1 + H * S1 * C_out;
    ASSERT_EQ(flops, expectedFLOPs);
}

} // namespace kas
