#pragma once

#include <limits>
#include <variant>

#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/NormalStage.hpp"


namespace kas {

class ReductionStage;

class ReductionStageStore {
    std::vector<std::unique_ptr<ReductionStage>> reductionStages;
public:
    template<typename... Args>
    ReductionStage *make(Args&&... args) {
        reductionStages.emplace_back(std::make_unique<ReductionStage>(std::forward<Args>(args)...));
        return reductionStages.back().get();
    }
};

class ReductionStage final: public AbstractStage {
    std::vector<const MapReduceOp *> reductions;

    NextSlotStore<NextOpSlot<MapReduceOp>> nextReductions;

    // The stage that is directly constructed, without appending any reduction.
    std::unique_ptr<NormalStage> nStage;

    void expand();

    void removeDeadChildrenFromSlots() override;
    void removeAllChildrenFromSlots() override;
    Finalizability checkForFinalizableChildren() const override;

    std::size_t uncheckedCountChildren();
    std::vector<Next> uncheckedGetChildrenHandles();
    const NextOpSlot<MapReduceOp> *getChildSlot(std::size_t key);
    std::optional<Node> uncheckedGetChild(Next next);
    std::optional<std::string> uncheckedGetChildDescription(Next next);

public:
    ReductionStage(ReductionStage& current, const MapReduceOp *nextReduction);
    // This is the root.
    ReductionStage(Sampler& sampler);

    const MapReduceOp *lastReduction() const { return reductions.size() ? reductions.back() : nullptr; }
    Interface toInterface() const;

    std::size_t hash() const override;
    std::size_t countChildren() override;
    std::vector<Next> getChildrenHandles() override;
    std::optional<Node> getChild(Next next) override;
    std::optional<std::string> getChildDescription(Next next) override;
    std::string description() const override;
};

} // namespace kas
