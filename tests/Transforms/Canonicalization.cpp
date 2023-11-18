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

} // namespace kas
