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
    void requestUpdateForFinalizability(bool propagate);

    // Python.
    virtual std::size_t hash() const = 0;
    virtual std::size_t countChildren() = 0;
    virtual std::vector<Next> getChildrenHandles() = 0;
    virtual std::optional<Node> getChild(Next next) = 0;
    virtual std::optional<std::string> getChildDescription(Next next) = 0;
    virtual std::string description() const = 0;

protected:
    // When the finalizability is determined, call parents to update their finalizability.
    void determineFinalizability(Finalizability yesOrNo, bool propagate);

    virtual void removeDeadChildrenFromSlots() = 0;
    virtual void removeAllChildrenFromSlots() = 0;
    virtual Finalizability checkForFinalizableChildren() const = 0;

    // A helper function that ensures the finalizability has been updated.
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
            if (counter > 100) {
                KAS_WARNING("guarded() is looping too many times.");
            }
        }
    }
};

} // namespace kas
