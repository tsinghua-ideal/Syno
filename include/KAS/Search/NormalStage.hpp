#pragma once

#include "KAS/Search/DimensionsStage.hpp"
#include "KAS/Search/Finalize.hpp"

namespace kas {

class NormalStage final: public DimensionsStage {
    // Lazily generate children.
    bool childrenGenerated = false;
    bool generatingChildren = false;

    GenericNextSlotStore<NextFinalizeSlot> nextFinalizations;

    void removeDeadChildrenFromSlots() override;
    void removeAllChildrenFromSlots() override;
    Finalizability checkForFinalizableChildren() const override;

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guardGeneratedChildren();

    std::shared_ptr<TensorView> getFinalize(const FinalizeOp *op) const;

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
        std::convertible_to<std::invoke_result_t<FP, const NextDimensionsStageSlot&>, R>
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
    NormalStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp);

    std::size_t countChildren() override;
    std::vector<Next> getChildrenHandles() override;
    std::vector<Arc> getChildrenArcs() override;
    std::optional<Arc> getArcFromHandle(Next next) override;
    std::optional<Node> getChild(Next next) override;
    bool canAcceptArc(Arc arc) override;
    Node getChild(Arc arc) override;
};

} // namespace kas
