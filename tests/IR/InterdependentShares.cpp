#include "Prelude.hpp"


namespace kas {

TEST(ir_tests, interdependent_shares) {
    auto ctx = BindingContext({"N=1", "H=16", "W=16"}, {"S=2"});
    BindingContext::DebugPublicCtx = &ctx;
    Forward::Factory factory { ctx };
    auto [sizeH, sizeW, sizeS] = factory.getSizes("H", "W", "S");

    auto [dimH_over_S, dimH_dot_S, dimW_dot_S, dimW_over_S] = factory.makeDimsOfSizes(sizeH / sizeS, sizeS, sizeS, sizeW / sizeS);
    // input: [H/S, S, W/S, S]
    auto [dimW_weight_1, dimH_over_S_weight_1] = factory.makeDimsOfSizes(sizeW, sizeH / sizeS);
    // weight_1: [W, H/S]
    auto [dimW_over_S_weight_2, dimH_weight_2] = factory.makeDimsOfSizes(sizeW / sizeS, sizeH);
    // weight_2: [W/S, H]

    // First Share.
    auto dimH_over_S_shared = Forward::ShareOp::Create(dimH_over_S, dimH_over_S_weight_1);
    auto dimW_over_S_shared = Forward::ShareOp::Create(dimW_over_S, dimW_over_S_weight_2);

    // Then Merge.
    auto dimH_merged = Forward::MergeOp::Create(dimH_over_S_shared, dimH_dot_S);
    auto dimW_merged = Forward::MergeOp::Create(dimW_over_S_shared, dimW_dot_S);

    // Share again.
    auto dimW_shared = Forward::ShareOp::Create(dimW_merged, dimW_weight_1);
    auto dimH_shared = Forward::ShareOp::Create(dimH_merged, dimH_weight_2);

    // Finally, output.
    dimW_shared.output(0);
    dimH_shared.output(1);

    auto topmosts = Forward::Factory::ForwardDimsToBackwardDims({
        {dimH_over_S, dimH_dot_S, dimW_dot_S, dimW_over_S},
        {dimW_weight_1, dimH_over_S_weight_1},
        {dimW_over_S_weight_2, dimH_weight_2},
    });
    auto tensorView = TensorView(topmosts, Parser("in_0 * in_1 * in_2").parseTensorExpression(), ctx);
    const IR& ir = tensorView.getSubgraphs();
    fmt::print("Result: {}\n", ir.outputTensor.toString(ctx));
    auto dfgGen = GraphvizDFGGen(ir, ctx);
    dfgGen.generate("./interdependent_shares.dot", "interdependent_shares");
    auto pytorchGen = PyTorchGen(ctx, tensorView);
    pytorchGen.generateSingle("./interdependent_shares.py", "InterdependentShares", tensorView, {});
    auto tvmGen = TVMCodeGen(ctx, ir);
    tvmGen.generate("./interdependent_shares_tvm.py");
}

} // namespace kas
