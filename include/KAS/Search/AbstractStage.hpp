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

    // Whether there exists a descendant of this Stage that can be Finalized.
    Finalizability finalizability = Finalizability::Maybe;

private:
    bool finalizabilityUpdateRequested = false;

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
        ChildrenFinalize,
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

    template<typename Op>
    int existingOp() const { return existingOps[Next::TypeOf<Op>()]; }

    Finalizability getFinalizability() const;
    void updateFinalizabilityOnRequest();

protected:
    // When the finalizability is determined, call parents to update their finalizability.
    void determineFinalizability(Finalizability yesOrNo);

    virtual void removeDeadChildrenFromSlots() = 0;
    virtual void removeAllChildrenFromSlots() = 0;
    virtual Finalizability checkForFinalizableChildren() const = 0;

    // A helper function that ensures the finalizability has been updated.
    template<typename F>
    auto guarded(F&& f) -> decltype(f()) {
        while (true) {
            if (finalizabilityUpdateRequested) {
                updateFinalizabilityOnRequest();
                finalizabilityUpdateRequested = false;
            }
            auto ret = f();
            if (!finalizabilityUpdateRequested) {
                return ret;
            }
        }
    }

    // A child of this stage calls this to signal that its finalizability has been updated.
    void requestUpdateForFinalizability();
};

} // namespace kas
