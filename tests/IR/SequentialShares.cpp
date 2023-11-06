#include "Prelude.hpp"


namespace kas {

TEST(ir_tests, sequential_shares) {
    auto ctx = BindingContext({"N", "H", "W"}, {});
    ctx.debug();
    Forward::Factory factory { ctx };
    auto [sizeN, sizeH, sizeW] = factory.getSizes("N", "H", "W");

    auto [dimN, dimH_input, dimH_weight_1, dimH_weight_2] = factory.makeDimsOfSizes(sizeN, sizeH, sizeH, sizeH);
    // [N, H], [H], [H]

    auto dimH_share_1 = Forward::ShareOp::Create(dimH_input, dimH_weight_1);
    auto dimH_share_2 = Forward::ShareOp::Create(dimH_share_1, dimH_weight_2);

    // Finally, output.
    dimN.output(0);
    dimH_share_2.output(1);

    factory.inputs({
        {dimN, dimH_input}, {dimH_weight_1}, {dimH_weight_2},
    });
    auto tensorView = TensorView(factory.getInputs(), Parser("in_0 * in_1 * in_2").parseTensorExpression(), ctx);
    const IR& ir = tensorView.getSubgraphs();
    fmt::print("Result: {}\n", ir.outputTensor.toString(ctx));
    auto dfgGen = GraphvizDFGGen(ir, ctx);
    dfgGen.generate("./sequential_shares.dot", "sequential_shares");
}

} // namespace kas
