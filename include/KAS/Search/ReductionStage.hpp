#pragma once

#include <limits>
#include <variant>

#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/NormalStage.hpp"


namespace kas {

class ReductionStage final: public AbstractStage {
    std::vector<const MapReduceOp *> reductions;

    // Since the stages are unique, we do not need a shared common store to store ReductionStage's.
    std::vector<std::unique_ptr<ReductionStage>> nextReductionStages;
    template<typename... Args>
    ReductionStage *make(Args&&... args) {
        nextReductionStages.emplace_back(std::make_unique<ReductionStage>(std::forward<Args>(args)...));
        return nextReductionStages.back().get();
    }
    NextSlotStore<NextOpSlot<MapReduceOp>> nextReductions;

    // The stage that is directly constructed, without appending any reduction.
    std::unique_ptr<NormalStage> nStage;

    void expand();

    void removeDeadChildrenFromSlots() override;
    void removeAllChildrenFromSlots() override;
    Finalizability checkForFinalizableChildren() const override;

    const NextOpSlot<MapReduceOp> *getChildSlot(std::size_t key) const;

public:
    ReductionStage(ReductionStage& current, const MapReduceOp *nextReduction);
    // This is the root.
    ReductionStage(Sampler& sampler);

    const MapReduceOp *lastReduction() const { return reductions.size() ? reductions.back() : nullptr; }
    Dimensions toInterface() const;

    std::size_t hash() const override;
    std::size_t countChildren() override;
    std::vector<Next> getChildrenHandles() override;
    std::vector<Arc> getArcs() override;
    std::optional<Arc> getArcFromHandle(Next next) override;
    std::optional<Node> getChild(Next next) override;
    Node getChild(Arc arc) override;
    std::string getChildDescription(Arc arc) override;
    std::string description() const override;
};

} // namespace kas
