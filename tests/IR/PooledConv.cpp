#include "Prelude.hpp"


namespace kas {

TEST(ir_tests, pooled_conv) {
    auto ctx = BindingContext(
        {"N=1", "C_in=3", "C_out=16", "H=16", "W=16"},
        {"K=3", "S=2"}
    );
    BindingContext::DebugPublicCtx = &ctx;
    Forward::Factory factory { ctx };
    auto [sizeN, sizeCin, sizeCout, sizeH, sizeW, sizeK, sizeS] = factory.getSizes("N", "C_in", "C_out", "H", "W", "K", "S");

    auto [dimN, dimCin_input, dimH, dimW] = factory.makeDimsOfSizes(sizeN, sizeCin, sizeH, sizeW);
    // [N, C_in, H, W], the input.

    auto [dimCout, dimCin_filter, dimK1, dimK2] = factory.makeDimsOfSizes(sizeCout, sizeCin, sizeK, sizeK);
    // [C_out, C_in, K, K], the filter.

    // First perform pooling on spatial dimensions.
    auto [dimH_over_S, dimH_dot_S] = Forward::SplitOp::Create(dimH, sizeS);
    dimH_dot_S.reduce(Reduce::ReduceType::Mean);
    auto [dimW_over_S, dimW_dot_S] = Forward::SplitOp::Create(dimW, sizeS);
    dimW_dot_S.reduce(Reduce::ReduceType::Mean);

    // Then perform convolution.
    auto [dimH_over_SK, dimH_dot_K] = Forward::UnfoldOp::Create(dimH_over_S, sizeK);
    auto dimK1_shared = Forward::ShareOp::Create(dimH_dot_K, dimK1);
    dimK1_shared.reduce(Reduce::ReduceType::Sum);
    auto [dimW_over_SK, dimW_dot_K] = Forward::UnfoldOp::Create(dimW_over_S, sizeK);
    auto dimK2_shared = Forward::ShareOp::Create(dimW_dot_K, dimK2);
    dimK2_shared.reduce(Reduce::ReduceType::Sum);
    auto dimCin_shared = Forward::ShareOp::Create(dimCin_input, dimCin_filter);
    dimCin_shared.reduce(Reduce::ReduceType::Sum);

    // Next, unpool.
    auto dimS_on_H = Forward::ExpandOp::Create(factory, sizeS);
    auto dimH_output = Forward::MergeOp::Create(dimH_over_SK, dimS_on_H);
    auto dimS_on_W = Forward::ExpandOp::Create(factory, sizeS);
    auto dimW_output = Forward::MergeOp::Create(dimW_over_SK, dimS_on_W);

    // Finally, output.
    dimN.output(0);
    dimCout.output(1);
    dimH_output.output(2);
    dimW_output.output(3);

    auto topmosts = Forward::Factory::ForwardDimsToBackwardDims({
        {dimN, dimCin_input, dimH, dimW, dimS_on_H, dimS_on_W},
        {dimCout, dimCin_filter, dimK1, dimK2},
    });
    auto tensorView = TensorView(topmosts, Parser("in_0 * in_1").parseTensorExpression(), ctx);
    auto graphvizGen = GraphvizGen(tensorView, ctx);
    graphvizGen.generate("./pooled_conv.dot", "pooled_conv");

    const IR& ir = tensorView.getSubgraphs();
    fmt::print("Result: {}\n", ir.outputTensor.toString(ctx));
    auto dfgGen = GraphvizDFGGen(ir, ctx);
    dfgGen.generate("./pooled_conv_dfg.dot", "pooled_conv");

    auto pytorchGen = PyTorchGen(ctx, tensorView);
    pytorchGen.generateSingle("./pooled_conv.py", "PooledConv", tensorView, {});

    auto tvmGen = TVMCodeGen(ctx, ir);
    tvmGen.generate("./pooled_conv_tvm.py");
}

} // namespace kas
