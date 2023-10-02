#include "Prelude.hpp"


namespace kas {

TEST(ir_tests, outer_product) {
    auto ctx = BindingContext({"N=1", "H=16", "W=16"}, {"S=2"});
    BindingContext::DebugPublicCtx = &ctx;
    Forward::Factory factory { ctx };
    auto [sizeN, sizeS] = factory.getSizes("N", "S");

    auto [dimN, dimS] = factory.makeDimsOfSizes(sizeN, sizeS);
    auto dimS_expanded = Forward::ExpandOp::Create(factory, sizeS);
    auto dimS_shared = Forward::ShareOp::Create(dimS_expanded, dimS);

    dimN.output(0);
    dimS_shared.output(1);

    auto topmosts = Forward::Factory::ForwardDimsToBackwardDims({
        {dimN, dimS_expanded}, {dimS},
    });
    auto tensorView = TensorView(topmosts, Parser("in_0 * in_1").parseTensorExpression(), ctx);
    const IR& ir = tensorView.getSubgraphs();
    fmt::print("Result: {}\n", ir.outputTensor.toString(ctx));
    auto dfgGen = GraphvizDFGGen(ir, ctx);
    dfgGen.generate("./outer_product.dot", "outer_product");
    auto pytorchGen = PyTorchGen(ctx, tensorView);
    pytorchGen.generateSingle("./outer_product.py", "OuterProduct", tensorView, {});
    auto tvmGen = TVMCodeGen(ctx, ir);
    tvmGen.generate("./outer_product_tvm.py");
}

} // namespace kas
