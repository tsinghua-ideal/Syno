#pragma once

#include "KAS/Search/Node.hpp"


namespace kas {

class Sampler;

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
    Finalizability state = Finalizability::Maybe;
    bool updateRequested = false;

    // When executing guarded(), we do not allow propagating over this Stage. In this case, requestUpdateForFinalizability() only sets the state to UpdateRequested.
    bool guardLock = false;

    bool isLocked() const { return guardLock; }

protected:
    // When locked, disallow any updateFinalizabilityOnRequest(). When unlocked, call updateFinalizabilityOnRequest() if needed.
    class FinalizabilityStateGuard {
        friend AbstractStage;
        AbstractStage *stage = nullptr;
        FinalizabilityStateGuard(AbstractStage *stage): stage { stage } {
            KAS_ASSERT(!stage->guardLock, "Cannot guard a stage twice!");
            stage->guardLock = true;
        }
    public:
        FinalizabilityStateGuard(const FinalizabilityStateGuard&) = delete;
        FinalizabilityStateGuard(FinalizabilityStateGuard&& other): stage { other.stage } {
            other.stage = nullptr;
        }
        void releaseAndPropagateChanges() {
            KAS_ASSERT(stage, "Guard invalid!");
            stage->guardLock = false;
            KAS_DEFER { stage = nullptr; };
            stage->updateFinalizabilityIfRequested();
        }
        ~FinalizabilityStateGuard() {
            if (!stage) return;
            KAS_ASSERT(stage->guardLock, "guardLock can only be modified by StateGuard!");
            stage->guardLock = false;
        }
    };
    [[nodiscard]] FinalizabilityStateGuard acquireFinalizabilityLock() {
        return FinalizabilityStateGuard { this };
    }

    // When the finalizability is determined, call parents to update their finalizability.
    void determineFinalizability(Finalizability yesOrNo);

    // When accessing any property used by the following function (from other functions), you must acquireFinalizabilityLock()!
    virtual void removeDeadChildrenFromSlots() = 0;
    // When accessing any property used by the following function (from other functions), you must acquireFinalizabilityLock()!
    virtual void removeAllChildrenFromSlots() = 0;
    // When accessing any property used by the following function (from other functions), you must acquireFinalizabilityLock()!
    virtual Finalizability checkForFinalizableChildren() const = 0;

private:
    // This tries to acquire the lock (to verify correctness). Do not call this if isLocked()!
    void updateFinalizabilityIfRequested();

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

    // Compute from Sampler.
    std::size_t remainingDepth() const;

    template<PrimitiveOpImpl Op>
    int existingOp() const { return existingOps[Next::TypeOf<Op>()]; }

    Finalizability getFinalizability() const;

    // A child of this stage calls this to signal that its finalizability has been updated.
    void requestUpdateForFinalizability();

    // Python.
    virtual std::size_t hash() const = 0;
    virtual std::size_t countChildren() = 0;
    virtual std::vector<Next> getChildrenHandles() = 0;
    virtual std::vector<Arc> getArcs() = 0;
    virtual std::optional<Arc> getArcFromHandle(Next next) = 0;
    virtual std::optional<Node> getChild(Next next) = 0;
    virtual Node getChild(Arc arc) = 0;
    virtual std::string description() const = 0;

    virtual ~AbstractStage() = default;
};

} // namespace kas
