#pragma once

#include "KAS/Search/AbstractStage.hpp"


namespace kas {

class NormalStage final: public AbstractStageBase<NormalStage> {
    friend class AbstractStageBase<NormalStage>;
    friend struct FinalStage;

    NodeType origin;

    // Lazily generate children.
    bool childrenGenerated = false;
    bool generatingChildren = false;

    NextFinalizeSlotStore nextFinalizations;
    using CollectedFinalizabilities = Base::CollectedFinalizabilities;
    void removeDeadChildrenFromSlots(const CollectedFinalizabilities& collected);
    void removeAllChildrenFromSlots();
    Finalizability checkForFinalizableChildren(const CollectedFinalizabilities& collected) const;

    void removeTooLongChains(ContractionOp::Analysis& analysis, const Graph& graph) const;
    Size getAllowanceUsage(const Graph& graph) const;

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guardGeneratedChildren();

    std::unique_ptr<FinalStage> getFinalize(const FinalizeOp& op);

    // If the required steps computed by possibleToFinalizeByExperimenting() is exactly the remaining depth, then we are in critical state.
    // In critical state, do not generate useless Ops.
    mutable ShapeDistance shapeDistance = ShapeDistance::Infinity;
    // This is for pruning. We experimentally finalize this stage, and conservatively exclude the stage if it is not possible to finalize.
    bool possibleToFinalizeByExperimenting() const override;

    std::size_t uncheckedCountChildren() const;
    std::vector<Next> uncheckedGetChildrenHandles() const;
    std::vector<Arc> uncheckedGetChildrenArcs() const;
    const NextFinalizeSlot *getChildFinalizeSlot(Next next) const;

    template<typename F>
    inline auto guarded(F&& f) -> decltype(f()) {
        guardGeneratedChildren();
        return f();
    }

    template<typename R, typename FP, typename FF>
    requires
        std::convertible_to<std::invoke_result_t<FF, const NextFinalizeSlot&>, R> &&
        std::convertible_to<std::invoke_result_t<FP, const NextStageSlot&>, R>
    std::optional<R> findTransform(Next next, FP&& fp, FF&& ff) const {
        switch (next.type) {
        case Next::Type::Finalize: {
            auto res = getChildFinalizeSlot(next);
            if (!res) return std::nullopt;
            return std::optional<R>(std::in_place, std::invoke(std::forward<FF>(ff), *res));
        }
        default: {
            return nextSlotStore.findTransform<R>(next, std::forward<FP>(fp));
        }
        }
    }

public:
    KAS_STATISTICS_DEF(
        ChildrenFinalize,
        FinalizabilityCheckInvocations,
        TooManyWeights,
        SharesUncanonical,
        ShapeDeviatesTooMuch,
    );
    // NormalStage cannot be root.
    NormalStage(GraphHandle interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock);
    // Currently there are 3 types of NormalStage.
    // 1. Embedded in ReductionStage. Constructed with deltaOp == std::nullopt. In this case this function returns the ReductionStage rather than the direct parent, i.e., a NormalStage.
    // 2. Child of ContractionStage. Constructed with deltaOp == Next::Type::Contraction. In this case we also just return the NormalStage because ContractionStage is simulated.
    // 3. Child of NormalStage. Others.
    bool isEmbeddedInReductionStage() const { return origin == NodeType::Reducing; }
    bool isFromContractionStage() const { return origin == NodeType::Contraction; }

    Finalizability experimentFinalizability(Lock& lock);
    ShapeDistance getShapeDistanceImpl() const;

    std::size_t countChildrenImpl();
    std::vector<Next> getChildrenHandlesImpl();
    std::vector<Arc> getChildrenArcsImpl();
    std::optional<Arc> getArcFromHandleImpl(Next next);
    std::optional<Node> getChildImpl(Next next);
    bool canAcceptArcImpl(Arc arc);
    std::optional<Node> getChildImpl(Arc arc);
};

} // namespace kas
