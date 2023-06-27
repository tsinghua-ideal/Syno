#pragma once

#include "KAS/Search/Node.hpp"


namespace kas {

class Sampler;
class ReductionStage;

class AbstractStage {
public:
    enum class Finalizability {
        Maybe, // Default state.
        Yes, // Determined by expanding subtrees, and see if there are final nodes.
        No, // Determined by expanding all subtrees or conservative experiments.
    };

protected:
    Sampler& sampler;

    // Parents of this Stage.
    std::vector<AbstractStage *> parents;

    // Stages with identical interfaces must be of the same depth.
    std::size_t depth;

    // A statistics counter for all Ops.
    Next::OpTypeCounter existingOps {};

private:
    enum class FinalizabilityState {
        Maybe,
        LocalYes, LocalNo,
        PropagatedYes, PropagatedNo,
    };
    FinalizabilityState state = FinalizabilityState::Maybe;
    bool isFinalizabilityDetermined() const { return state != FinalizabilityState::Maybe; }
    bool isFinalizabilityDeterminedButNotPropagated() const { return state == FinalizabilityState::LocalYes || state == FinalizabilityState::LocalNo; }
    bool isFinalizabilityDeterminedAndPropagated() const { return state == FinalizabilityState::PropagatedYes || state == FinalizabilityState::PropagatedNo; }
    static constexpr Finalizability GetFinalizabilityFromState(FinalizabilityState state) {
        switch (state) {
        case FinalizabilityState::Maybe: return Finalizability::Maybe;
        case FinalizabilityState::LocalYes: return Finalizability::Yes;
        case FinalizabilityState::LocalNo: return Finalizability::No;
        case FinalizabilityState::PropagatedYes: return Finalizability::Yes;
        case FinalizabilityState::PropagatedNo: return Finalizability::No;
        default: KAS_UNREACHABLE();
        }
    }

    enum class ConstructionState {
        InitialConstruction,
        Normal,
        InConstruction,
    };
    ConstructionState constructionState = ConstructionState::InitialConstruction;
    bool inConstruction() const { return constructionState != ConstructionState::Normal; }

protected:
    // Call this at the end of constructor!
    void finishInitialConstruction() {
        KAS_ASSERT(constructionState == ConstructionState::InitialConstruction);
        constructionState = ConstructionState::Normal;
        updateFinalizability();
    }

    template<typename F>
    void construct(F&& f) {
        KAS_ASSERT(constructionState != ConstructionState::InitialConstruction, "You forgot to call finishInitialConstruction()");
        KAS_ASSERT(constructionState != ConstructionState::InConstruction, "construct() is called recursively!");
        constructionState = ConstructionState::InConstruction;
        KAS_DEFER {
            KAS_ASSERT(constructionState == ConstructionState::InConstruction, "constructionState is mistakenly unset!");
            constructionState = ConstructionState::Normal;
            // Note that if the following line throws, the program terminates.
            // This is desired. Because we do not want the program in inconsistent state.
            updateFinalizability();
        };
        f();
    }

    // When the finalizability is determined, call parents to update their finalizability.
    // When in construction, this function does not call the update. Otherwise, update is called.
    void determineFinalizability(Finalizability yesOrNo);

    // When accessing any property used by the following function (from other functions), you must wrap it in construct()! Before this is called, it is asserted that we are not in construction.
    virtual void removeDeadChildrenFromSlots() = 0;
    // When accessing any property used by the following function (from other functions), you must wrap it in construct()! Before this is called, it is asserted that we are not in construction.
    virtual void removeAllChildrenFromSlots() = 0;
    // When accessing any property used by the following function (from other functions), you must wrap it in construct()! Before this is called, it is asserted that we are not in construction.
    virtual Finalizability checkForFinalizableChildren() const = 0;

private:
    // Update finalizability, if not in construction.
    // If the state is not propagated, then notify the parents.
    void updateFinalizability();

public:
    KAS_STATISTICS_DEF(
        Creations,
        ChildrenMapReduce,
        ChildrenShift,
        ChildrenStride,
        ChildrenSplit,
        ChildrenUnfold,
        ChildrenMerge,
        ChildrenShare,
        FinalizabilityMaybe,
        FinalizabilityYes,
        FinalizabilityNo,
    )
    // Create a root stage.
    AbstractStage(Sampler& sampler);
    // Create a non-root stage.
    AbstractStage(AbstractStage& creator, std::optional<Next::Type> deltaOp);
    void addParent(AbstractStage& parent);

    // Disallow copy or move.
    AbstractStage(const AbstractStage&) = delete;
    AbstractStage(AbstractStage&&) = delete;

    virtual const Dimensions& getInterface() const = 0;

    // Compute from Sampler.
    std::size_t remainingDepth() const;

    template<PrimitiveOpImpl Op>
    int existingOp() const { return existingOps[Next::TypeOf<Op>()]; }

    // When in construction, this function returns Maybe. Otherwise, it returns the state.
    Finalizability getFinalizability() const;

    // Python.
    virtual std::size_t hash() const = 0;
    virtual std::size_t countChildren() = 0;
    virtual std::vector<Next> getChildrenHandles() = 0;
    virtual std::vector<Arc> getChildrenArcs() = 0;
    virtual std::optional<Arc> getArcFromHandle(Next next) = 0;
    virtual std::optional<Node> getChild(Next next) = 0;
    virtual bool canAcceptArc(Arc arc) = 0;
    virtual Node getChild(Arc arc) = 0;
    virtual std::string description() const = 0;

    virtual ~AbstractStage() = default;
};

} // namespace kas
