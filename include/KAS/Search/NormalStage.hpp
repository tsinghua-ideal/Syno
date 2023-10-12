#pragma once

#include "KAS/Search/AbstractStage.hpp"


namespace kas {

class NormalStage final: public AbstractStageBase<NormalStage> {
    friend class AbstractStageBase<NormalStage>;

    // Lazily generate children.
    bool childrenGenerated = false;
    bool generatingChildren = false;

    GenericNextSlotStore<NextFinalizeSlot> nextFinalizations;
    using Base::CollectedFinalizabilities;
    void removeDeadChildrenFromSlots(const CollectedFinalizabilities& collected);
    void removeAllChildrenFromSlots();
    Finalizability checkForFinalizableChildren(const CollectedFinalizabilities& collected) const;

    GraphHandle removeTooLongChains(const Graph& graph, const GraphHandle& interface) const;

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guardGeneratedChildren();

    std::unique_ptr<TensorView> getFinalize(const FinalizeOp& op) const;

    // If the required steps computed by possibleToFinalizeByExperimenting() is exactly the remaining depth, then we are in critical state.
    // In critical state, do not generate useless Ops.
    mutable bool inCriticalState = false;
    // This is for pruning. We experimentally finalize this stage, and conservatively exclude the stage if it is not possible to finalize.
    bool possibleToFinalizeByExperimenting() const;

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
        ShapeDeviatesTooMuch,
    );
    // NormalStage cannot be root.
    NormalStage(GraphHandle interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock);

    std::size_t countChildrenImpl();
    std::vector<Next> getChildrenHandlesImpl();
    std::vector<Arc> getChildrenArcsImpl();
    std::optional<Arc> getArcFromHandleImpl(Next next);
    std::optional<Node> getChildImpl(Next next);
    bool canAcceptArcImpl(Arc arc);
    Node getChildImpl(Arc arc);
};

} // namespace kas
