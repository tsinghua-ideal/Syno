#pragma once

#include <array>
#include <iterator>

#include "KAS/Search/Finalize.hpp"
#include "KAS/Transforms.hpp"


namespace kas {

struct StatisticsCollector {
    static inline void PrintSummary(std::ostream& os) {
        auto it = [&]() {
            return std::ostreambuf_iterator<char>(os);
        };

        fmt::format_to(it(), "Summary for color semantics:\n");
        struct Stats {
            const char *name;
            std::size_t trials;
            std::size_t successes;
            float rate() const noexcept { return static_cast<float>(successes) / trials; }
        };
        constexpr std::size_t OpCnt = 6;
        Stats stats[OpCnt] = {
            { "Merge", MergeOp::CountColorTrials, MergeOp::CountColorSuccesses },
            { "Share", ShareOp::CountColorTrials, ShareOp::CountColorSuccesses },
            { "Shift", ShiftOp::CountColorTrials, ShiftOp::CountColorSuccesses },
            { "Split", SplitOp::CountColorTrials, SplitOp::CountColorSuccesses },
            { "Stride", StrideOp::CountColorTrials, StrideOp::CountColorSuccesses },
            { "Unfold", UnfoldOp::CountColorTrials, UnfoldOp::CountColorSuccesses },
        };
        fmt::format_to(it(), "  Success rates:\n");
        std::size_t trials = 0;
        for (std::size_t i = 0; i < OpCnt; ++i) {
            fmt::format_to(it(), "    {}: {:.2f} ({} / {})\n", stats[i].name, stats[i].rate(), stats[i].successes, stats[i].trials);
            trials += stats[i].trials;
        }
        fmt::format_to(it(), "  Inconsistent colors:\n");
        std::size_t incon = Colors::CountColorInconsistent;
        fmt::format_to(it(), "    {:.2f} ({} / {})\n", static_cast<float>(incon) / trials, incon, trials);

        fmt::format_to(it(), "Summary for finalizations:\n");
        fmt::format_to(it(), "  CountSuccesses: {}\n", FinalizeOp::CountSuccesses);
        fmt::format_to(it(), "  CountFailures: {}\n", FinalizeOp::CountFailures);
        fmt::format_to(it(), "  CountLegalFinalizations: {}\n", FinalizeOp::CountLegalFinalizations);
        fmt::format_to(it(), "  CountConflictingColors: {}\n", FinalizeOp::CountConflictingColors);
    }
};

} // namespace kas
