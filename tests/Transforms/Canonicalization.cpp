#include "Prelude.hpp"


namespace kas {

TEST_F(transforms_tests, unorderedness_shift) {
    auto shiftC = ShiftOp(&itC, 1);
    Graph::DimensionMap<std::size_t> unordered {{shiftC.getInput(), 0}};
    auto canonicalizer = UnorderednessCanonicalizer(unordered);
    const Graph graph = GraphHandle({shiftC.getInput()}, {}).buildGraph();
    graph.accept(canonicalizer);
    ASSERT_TRUE(canonicalizer.uncanonical);
}

TEST_F(transforms_tests, unorderedness_iterator) {
    auto mergeH = MergeOp(&itH, sizeC);
    auto splitH = SplitOp(mergeH.getInputL(), mergeH.getInputR());
    Graph::DimensionMap<std::size_t> unordered {{splitH.getInput(), 0}};
    auto canonicalizer = UnorderednessCanonicalizer(unordered);
    const Graph graph = GraphHandle({splitH.getInput()}, {}).buildGraph();
    graph.accept(canonicalizer);
    ASSERT_TRUE(canonicalizer.uncanonical);
}

TEST_F(transforms_tests, unorderedness_reduce) {
    Dimension rH = reduceH.getInput(0), rC = reduceC.getInput(0);
    auto splitCH = SplitOp(rH, rC);
    Graph::DimensionMap<std::size_t> unordered {{splitCH.getInput(), 0}};
    auto canonicalizer = UnorderednessCanonicalizer(unordered);
    const Graph graph = GraphHandle({splitCH.getInput()}, {}).buildGraph();
    graph.accept(canonicalizer);
    ASSERT_TRUE(canonicalizer.at(rH).sourceSplitOp);
    ASSERT_TRUE(canonicalizer.at(rC).sourceSplitOp);
    ASSERT_EQ(canonicalizer.at(rH).sourceSplitOp, canonicalizer.at(rC).sourceSplitOp);
}

TEST_F(transforms_tests, unordered_split) {
    auto test = [&](const SplitOp& split) {
        Graph::DimensionMap<std::size_t> unordered {{split.getInput(), 0}};
        auto canonicalizer = UnorderednessCanonicalizer(unordered);
        const Graph graph = GraphHandle({split.getInput()}, {}).buildGraph();
        graph.accept(canonicalizer);
        return canonicalizer.uncanonical;
    };
    bool uncanonicalHW = false, uncanonicalWH = false;
    {
        auto split = SplitOp(&itH, &itW);
        uncanonicalHW = test(split);
    }
    {
        auto split = SplitOp(&itW, &itH);
        uncanonicalWH = test(split);
    }
    ASSERT_EQ(uncanonicalHW + uncanonicalWH, 1);
}

} // namespace kas
