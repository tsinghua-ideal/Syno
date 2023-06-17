#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Search/Sample.hpp"

namespace kas {

AbstractStage::AbstractStage(Sampler& sampler):
    sampler { sampler },
    parents {},
    depth { 0 },
    existingOps {}
{}

AbstractStage::AbstractStage(AbstractStage& creator, std::optional<Next::Type> optionalDeltaOp):
    sampler { creator.sampler },
    depth { creator.depth + static_cast<std::size_t>(optionalDeltaOp.has_value()) },
    existingOps { creator.existingOps }
{
    ++CountCreations;
    parents.emplace_back(&creator);
    if (optionalDeltaOp) {
        Next::Type deltaOp = *optionalDeltaOp;
        existingOps[deltaOp] += 1;
        switch (deltaOp) {
        case Next::Type::MapReduce: ++CountChildrenMapReduce; break;
        case Next::Type::Shift: ++CountChildrenShift; break;
        case Next::Type::Stride: ++CountChildrenStride; break;
        case Next::Type::Split: ++CountChildrenSplit; break;
        case Next::Type::Unfold: ++CountChildrenUnfold; break;
        case Next::Type::Merge: ++CountChildrenMerge; break;
        case Next::Type::Share: ++CountChildrenShare; break;
        case Next::Type::Finalize: ++CountChildrenFinalize; break;
        default: KAS_UNREACHABLE();
        }
    }
    ++CountFinalizabilityMaybe;
}

void AbstractStage::addParent(AbstractStage& parent) {
    parents.emplace_back(&parent);
}

std::size_t AbstractStage::remainingDepth() const {
    const std::size_t maxDepth = sampler.getOptions().depth;
    KAS_ASSERT(maxDepth >= depth);
    return maxDepth - depth;
}

AbstractStage::Finalizability AbstractStage::getFinalizability() const {
    return finalizability;
}

void AbstractStage::determineFinalizability(Finalizability yesOrNo) {
    KAS_ASSERT(finalizability == Finalizability::Maybe, "Finalizability has already been determined.");
    switch (yesOrNo) {
    case Finalizability::Yes:
        --CountFinalizabilityMaybe;
        ++CountFinalizabilityYes;
        finalizability = Finalizability::Yes;
        break;
    case Finalizability::No:
        --CountFinalizabilityMaybe;
        ++CountFinalizabilityNo;
        finalizability = Finalizability::No;
        break;
    default:
        KAS_CRITICAL("Invalid Finalizability.");
    }
    // Signal parents.
    for (AbstractStage *parent : parents) {
        parent->requestUpdateForFinalizability();
    }
}

void AbstractStage::updateFinalizabilityOnRequest() {
    // If finalizability is determined, still need to remove dead ends!
    if (getFinalizability() == Finalizability::Yes) {
        removeDeadChildrenFromSlots();
        return;
    } else if (getFinalizability() == Finalizability::No) {
        // Usually this is not needed, becase when we determined Finalizability::No, we would have removed all children.
        removeAllChildrenFromSlots();
        return;
    }

    // Next, we need to check if this is finalizable. If we determine the Finalizability, we must propagate it.
    Finalizability newFinalizability = checkForFinalizableChildren();
    if (newFinalizability != Finalizability::Maybe) {
        determineFinalizability(newFinalizability);
    }
    removeDeadChildrenFromSlots();
    return;
}

void AbstractStage::requestUpdateForFinalizability() {
    finalizabilityUpdateRequested = true;
}

} // namespace kas
