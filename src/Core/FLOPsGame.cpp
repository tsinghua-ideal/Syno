#include "KAS/Core/FLOPsGame.hpp"
#include "KAS/Utils/Algorithm.hpp"


namespace kas {

std::size_t FLOPsGame::State::getNextDecrease(std::size_t i) const {
    while (i < game.decrease.size() && decreaseDetermined(i)) {
        ++i;
    }
    return i;
}

FLOPsGame::State::State(const FLOPsGame& game):
    game{ game }, current { game.inputSize },
    determinedIncrease(game.increase.size()),
    determinedDecrease(game.decrease.size())
{}

std::size_t FLOPsGame::State::FLOPs() const {
    if (game.decrease.empty()) {
        // No reductions. Happy!
        return 0;
    } else {
        // We first check if there are reductions that do not depend on any increase, which is a rather common case.
        std::optional<State> remaining;
        for (std::size_t decrease = 0; decrease < game.decrease.size(); ++decrease) {
            if (std::ranges::all_of(game.dependencies[decrease], std::logical_not<>{})) {
                // This decrease does not depend on any increase. Do it first.
                if (!remaining) {
                    remaining.emplace(*this);
                }
                remaining->determinedDecrease[decrease] = true;
            }
        }
        // Start.
        if (remaining.has_value()) {
            return remaining->remainingFLOPs(*this);
        } else {
            return experimentRemainingFLOPs(*this, 0);
        }
    }
}

// This function is guaranteed to make progress.
// That is, we have done at least 1 reduction.
std::size_t FLOPsGame::State::remainingFLOPs(const State& base) const {
    // First count delta FLOPs.
    // `current` is the number of iterations.
    // Note that the common case is the contraction of 2 tensors.
    // So a multiplication is absorbed in FMA.
    // We only provide a conservative estimation, so this is enough.
    std::size_t deltaFLOPs = current.evalSumAllConsts(game.ctx);

    // Then find out unfinished reductions, i.e., decrease.
    const std::size_t nowDecreasing = getNextDecrease(0);

    // If all reductions are done, there are no more FLOPs.
    if (nowDecreasing == game.decrease.size()) {
        return deltaFLOPs;
    }

    // Start another experiment. We must finish the remaining decreases.
    auto remaining = *this;
    bool success = false;
    // Apply the reductions we have done in the round.
    for (std::size_t i = 0; i < game.decrease.size(); ++i) {
        if (!base.decreaseDetermined(i) && decreaseDetermined(i)) {
            // Newly done reduction.
            remaining.current /= game.decrease[i];
            success = true;
        }
    }
    KAS_ASSERT(success, "Invariant violated: FLOPsGame::State::remainingFLOPs makes no progress!");
    // Fold.
    return deltaFLOPs + remaining.experimentRemainingFLOPs(*this, nowDecreasing);
}

std::size_t FLOPsGame::State::experimentRemainingFLOPs(const State& base, std::size_t nowDecreasing) const {
    // First check if we have finished the experiment.
    if (nowDecreasing == game.decrease.size()) {
        // Yes, we have finished the experiment.
        // Check if we have done any reductions.
        if (base.determinedDecrease == determinedDecrease) {
            // No reductions made. This is not a valid trial.
            // Sanity check.
            KAS_ASSERT(base.determinedIncrease == determinedIncrease, "No, increasing numel without any reductions is a waste of FLOPs.");
            return Infinity;
        } else {
            // We have done some reductions. It is worthwhile to perform a trial.
            return remainingFLOPs(base);
        }
    }

    // Start experimenting.
    std::size_t bestFLOPs = Infinity;

    // Then enumerate trials.
    KAS_ASSERT(!decreaseDetermined(nowDecreasing));
    // First is the trial with this reduced.
    {
        auto trial = *this;
        trial.determinedDecrease[nowDecreasing] = true;
        for (std::size_t increase = 0; increase < game.increase.size(); ++increase) {
            if (
                // This decrease depends on this increase.
                game.dependencies[nowDecreasing][increase]
                // And this increase is not determined.
                && !determinedIncrease[increase]
            ) {
                // This is a new increase!
                trial.determinedIncrease[increase] = true;
                trial.current *= game.increase[increase];
            }
        }
        // Check if some of the added dependencies requires compulsory reductions before nowDecreasing. If so, we could have done that before, so this is is not needed. If the compulsory reductions are all after nowDecreasing, do them right now.
        for (std::size_t decrease = 0; decrease < game.decrease.size(); ++decrease) {
            if (
                // The decrease is newly made available.
                !trial.decreaseDetermined(decrease)
                && std::ranges::all_of(
                    std::views::iota(0_uz, game.increase.size()),
                    [&](std::size_t increase) -> bool {
                        return !game.dependencies[decrease][increase] || trial.determinedIncrease[increase];
                    }
                )
            ) {
                if (decrease < nowDecreasing) {
                    // This is not even necessary.
                    return Infinity;
                } else if (decrease > nowDecreasing) {
                    // Do them right now.
                    trial.determinedDecrease[decrease] = true;
                } else {
                    KAS_UNREACHABLE();
                }
            }
        }
        const std::size_t FLOPsWithThis = trial.experimentRemainingFLOPs(base, trial.getNextDecrease(nowDecreasing + 1));
        bestFLOPs = std::min(bestFLOPs, FLOPsWithThis);
    }
    // Then is the trial without this reduced.
    {
        const std::size_t FLOPsWithoutThis = experimentRemainingFLOPs(base, getNextDecrease(nowDecreasing + 1));
        bestFLOPs = std::min(bestFLOPs, FLOPsWithoutThis);
    }
    return bestFLOPs;
}

std::size_t FLOPsGame::FLOPs() const {
    auto state = State(*this);
    return state.FLOPs();
}

} // namespace kas
