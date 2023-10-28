#include <fmt/ranges.h>

#include <gtest/gtest.h>
#include <ranges>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Statistics.hpp"


using namespace kas;

TEST(serach_misc_tests, size_equals_one) {
    auto options = SampleOptions();
    options.depth = 5;
    options.maximumFinalizations = 1;
    options.maximumExpands = 1;
    options.maximumMerges = 1;
    options.maximumReductions = 1;
    options.maximumShares = 1;
    options.maximumShifts = 0;
    options.maximumSplits = 1;
    options.maximumStrides = 0;
    options.maximumUnfolds = 0;
    options.maximumEnumerationsPerVar = 3;
    options.maximumTensors = 2;
    auto sampler = Sampler("[C_in]", "[s^-1*C_out]", {"C_in=64:1", "C_out=64:2"}, {"s=2:2", "g=32:3"}, {{}}, {}, options, 16);
    auto& ctx = sampler.getBindingContext();
    ctx.debug();
    auto root = sampler.visit({}).value();
    root.expandSync(6);
    sampler.getPruner().sync();
    auto node = root
        .getChild(Next(Next::TypeOf<ReduceOp>(), 15623912910957193693_uz)).value()
        .getChild(Next(Next::TypeOf<MergeOp>(), 3937223695884881477_uz)).value()
        .getChild(Next(Next::TypeOf<SplitOp>(), 15362865786984789864_uz)).value()
        .getChild(Next(Next::TypeOf<ShareOp>(), 10573079388359089061_uz)).value()
        .getChild(Next(Next::TypeOf<ExpandOp>(), 9719963751023831356_uz)).value()
        .getChild(Next(Next::Type::Finalize, 7705279065707787209_uz)).value();
    fmt::print("Node {}:\n{}\n", node.toString(), GraphvizDFGGen::Print(node.asFinalStage()->value.getSubgraphs(), ctx));
    ASSERT_TRUE(!root.isDeadEnd());
    StatisticsCollector::PrintSummary(std::cout);
}
