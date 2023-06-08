#pragma once

#include <limits>
#include <variant>

#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/Stage.hpp"


namespace kas {

class ReductionStage {
public:
    static constexpr std::size_t StopReductionToken = std::numeric_limits<std::size_t>::max();
    struct NextReductionSlot: NextSlot<Next::Type::MapReduce> {
        std::unique_ptr<ReductionStage> next;
        static std::size_t GetKey(const MapReduceOp *op) { return op->hash(); }
    };

private:
    Sampler& sampler;

    std::vector<const MapReduceOp *> reductions;

    NextSlotStore<NextReductionSlot> nextReductions;

    // The stage that is directly constructed, without appending any reduction.
    std::unique_ptr<Stage> stage;

    ReductionStage(Sampler& sampler, std::vector<const MapReduceOp *>&& reductions);

public:
    ReductionStage(const ReductionStage& current, const MapReduceOp *nextReduction):
        ReductionStage(current.sampler, [&]{
            auto newReductions = current.reductions;
            newReductions.emplace_back(nextReduction);
            return newReductions;
        }()) {}
    // This is the root.
    ReductionStage(Sampler& sampler):
        ReductionStage(sampler, std::vector<const MapReduceOp *> {}) {}

    std::size_t hash() const;

    const MapReduceOp *lastReduction() const { return reductions.size() ? reductions.back() : nullptr; }
    Interface toInterface() const;

    std::size_t countChildren() const { return nextReductions.size() + 1; }
    // Aside from slots, return a special Next (with key StopReductionToken).
    std::vector<Next> getChildrenHandles() const;
    const NextReductionSlot& getChildSlot(std::size_t key) const;
    // Use a special Next (with key StopReductionToken) to transition to the normal Stage.
    Node getChild(Next next) const;
    std::string getChildDescription(std::size_t key) const;
    std::string description() const;
};

} // namespace kas
