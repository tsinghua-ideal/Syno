#pragma once

#include <array>
#include <iterator>

#include "KAS/Search/Finalize.hpp"
#include "KAS/Transforms.hpp"


namespace kas {

struct StatisticsCollector {
    static void PrintSummary(std::ostream& os) {
        auto it = [&]() {
            return std::ostreambuf_iterator<char>(os);
        };
#define COLLECT_STATS_FOR_OP(Op) \
        fmt::format_to(it(), "Summary for " #Op ":\n"); \
        Op::PrintStatistics(os);
        COLLECT_STATS_FOR_OP(MergeOp)
        COLLECT_STATS_FOR_OP(ShareOp)
        COLLECT_STATS_FOR_OP(SplitOp)
        COLLECT_STATS_FOR_OP(StrideOp)
        COLLECT_STATS_FOR_OP(UnfoldOp)
        COLLECT_STATS_FOR_OP(FinalizeOp)
#undef COLLECT_STATS_FOR_OP
    }
};

} // namespace kas
