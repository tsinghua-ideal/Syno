#pragma once

#include <array>
#include <iterator>

#include "KAS/Core/IR.hpp"
#include "KAS/Core/Lower.hpp"
#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Transforms/Transforms.hpp"


namespace kas {

struct StatisticsCollector {
    static void PrintSummary(std::ostream& os) {
        auto it = [&]() {
            return std::ostreambuf_iterator<char>(os);
        };
#define KAS_COLLECT_STATS_FOR_OP(Op) \
        fmt::format_to(it(), "Summary for " #Op ":\n"); \
        Op::PrintStatistics(os);
        KAS_COLLECT_STATS_FOR_OP(DimensionEvaluator)
        KAS_COLLECT_STATS_FOR_OP(ExpandOp)
        KAS_COLLECT_STATS_FOR_OP(MergeOp)
        KAS_COLLECT_STATS_FOR_OP(ShiftOp)
        KAS_COLLECT_STATS_FOR_OP(SplitOp)
        KAS_COLLECT_STATS_FOR_OP(StrideOp)
        KAS_COLLECT_STATS_FOR_OP(UnfoldOp)
        KAS_COLLECT_STATS_FOR_OP(IR)
        KAS_COLLECT_STATS_FOR_OP(FinalizeOp)
        KAS_COLLECT_STATS_FOR_OP(AbstractStage)
        KAS_COLLECT_STATS_FOR_OP(NormalStage)
        KAS_COLLECT_STATS_FOR_OP(Node)
#undef KAS_COLLECT_STATS_FOR_OP
    }
};

} // namespace kas
