#pragma once

#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Transforms/MapReduce.hpp"


namespace kas {

class ReductionStage final: public AbstractStageBase<ReductionStage> {
    friend class AbstractStageBase<ReductionStage>;

    // The stage that is directly constructed, without appending any reduction.
    std::unique_ptr<NormalStage> nStage;

    void expand();

    struct CollectedFinalizabilities: Base::CollectedFinalizabilities {
        Finalizability nStageFinalizability;
    };
    CollectedFinalizabilities collectFinalizabilities();
    Finalizability checkForFinalizableChildren(const CollectedFinalizabilities& collected) const;

public:
    // This is the root.
    ReductionStage(Sampler& sampler, Dimensions interface, Lock lock);
    ReductionStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock);

    std::size_t countChildrenImpl();
    std::vector<Next> getChildrenHandlesImpl();
    std::vector<Arc> getChildrenArcsImpl();
    std::optional<Arc> getArcFromHandleImpl(Next next);
    std::optional<Node> getChildImpl(Next next);
    bool canAcceptArcImpl(Arc arc);
    Node getChildImpl(Arc arc);
};

} // namespace kas
