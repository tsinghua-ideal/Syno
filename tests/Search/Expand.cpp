#include "Prelude.hpp"


using namespace kas;

TEST_F(search_tests, expand) {
    // Time this.
    auto start = std::chrono::high_resolution_clock::now();
    auto root = sampler.visit({});
    sampler.getExpander().expandSync(*root, 4);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    StatisticsCollector::PrintSummary(std::cout);
    fmt::print("Time: {} ms\n", duration.count());
}
