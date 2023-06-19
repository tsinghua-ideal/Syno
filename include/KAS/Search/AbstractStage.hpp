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
    // Whether there exists a descendant of this Stage that can be Finalized.
    Finalizability finalizability = Finalizability::Maybe;

    bool finalizabilityUpdateRequested = false;

    void updateFinalizabilityOnRequest();

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

    template<typename Op>
    int existingOp() const { return existingOps[Next::TypeOf<Op>()]; }

    Finalizability getFinalizability() const;

    // A child of this stage calls this to signal that its finalizability has been updated.
    void requestUpdateForFinalizability();

    // Python.
    virtual std::size_t hash() const = 0;
    virtual std::size_t countChildren() = 0;
    virtual std::vector<Next> getChildrenHandles() = 0;
    virtual std::optional<Node> getChild(Next next) = 0;
    virtual std::optional<std::string> getChildDescription(Next next) = 0;
    virtual std::string description() const = 0;

    virtual ~AbstractStage() = default;

protected:
    // When the finalizability is determined, call parents to update their finalizability.
    void determineFinalizability(Finalizability yesOrNo);

    virtual void removeDeadChildrenFromSlots() = 0;
    virtual void removeAllChildrenFromSlots() = 0;
    virtual Finalizability checkForFinalizableChildren() const = 0;

    // A helper function that ensures the finalizability has been updated.
    // We want all the calls to be active. That is, if a child wants to propagate that it is dead, then it requests for update, rather than recursively calling parents to update. In this way we avoid conflicting access to data, and keep the control flow simple.
    // After calling the desired function, if this stage finds out that it is required to be updated, then this stage performs an update of finalizability, and do another trial.
    template<typename F>
    inline auto guarded(F&& f) -> decltype(f()) {
        int counter = 0;
        while (true) {
            if (finalizabilityUpdateRequested) {
                updateFinalizabilityOnRequest();
            }
            auto ret = f();
            if (!finalizabilityUpdateRequested) {
                return ret;
            }
            ++counter;
            if (counter > 1000) {
                KAS_CRITICAL("AbstractStage::guarded() is looping too many times. Check for cycles in the graph.");
            }
        }
    }
};

} // namespace kas
