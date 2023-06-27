#pragma once

#include "KAS/Search/DimensionsStage.hpp"
#include "KAS/Transforms/MapReduce.hpp"


namespace kas {

class ReductionStage final: public DimensionsStage {
    // The stage that is directly constructed, without appending any reduction.
    std::unique_ptr<NormalStage> nStage;

    void expand();

    void removeAllChildrenFromSlots() override;
    Finalizability checkForFinalizableChildren() const override;

public:
    // This is the root.
    ReductionStage(Sampler& sampler);
    ReductionStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp);

    std::size_t countChildren() override;
    std::vector<Next> getChildrenHandles() override;
    std::vector<Arc> getChildrenArcs() override;
    std::optional<Arc> getArcFromHandle(Next next) override;
    std::optional<Node> getChild(Next next) override;
    bool canAcceptArc(Arc arc) override;
    Node getChild(Arc arc) override;
};

} // namespace kas
