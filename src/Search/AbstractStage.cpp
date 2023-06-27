#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Search/Sample.hpp"

namespace kas {

void AbstractStage::determineFinalizability(Finalizability yesOrNo) {
    KAS_ASSERT(!isFinalizabilityDetermined(), "Finalizability has already been determined.");
    switch (yesOrNo) {
    case Finalizability::Yes:
        --CountFinalizabilityMaybe;
        ++CountFinalizabilityYes;
        state = FinalizabilityState::LocalYes;
        break;
    case Finalizability::No:
        --CountFinalizabilityMaybe;
        ++CountFinalizabilityNo;
        state = FinalizabilityState::LocalNo;
        break;
    default:
        KAS_CRITICAL("Invalid Finalizability.");
    }
    updateFinalizability();
}

void AbstractStage::updateFinalizability() {
    if (inConstruction()) {
        // If we are in construction, do nothing. Because we know that this function will later be called.
        return;
    }

    auto signalParents = [this] {
        if (state == FinalizabilityState::LocalYes) {
            state = FinalizabilityState::PropagatedYes;
        } else if (state == FinalizabilityState::LocalNo) {
            state = FinalizabilityState::PropagatedNo;
        }
        for (auto parent: parents) {
            parent->updateFinalizability();
        }
    };

    switch (state) {
    case FinalizabilityState::Maybe: {
        // First carry out check.
        Finalizability newFinalizability = checkForFinalizableChildren();
        if (newFinalizability == Finalizability::Maybe) {
            // Well, nothing new.
            // But we still need to remove the dead ends.
            removeDeadChildrenFromSlots();
        } else {
            // Otherwise, we have determined the finalizability.
            determineFinalizability(newFinalizability);
            // Since this function is called again, we do not need to propagate the change.
        }
        break;
    }
    case FinalizabilityState::LocalYes: {
        // Still need to remove dead ends.
        removeDeadChildrenFromSlots();
        // And we need to signal parents.
        signalParents();
        break;
    }
    case FinalizabilityState::LocalNo: {
        // Usually this is not needed, becase when we determined Finalizability::No, we would have removed all children.
        removeAllChildrenFromSlots();
        // And propagate.
        signalParents();
        break;
    }
    case FinalizabilityState::PropagatedYes: {
        // Remove dead ends and no need to propagate.
        removeDeadChildrenFromSlots();
        break;
    }
    case FinalizabilityState::PropagatedNo: {
        // Remove all children and no need to propagate.
        removeAllChildrenFromSlots();
        break;
    }
    }
}

AbstractStage::AbstractStage(Sampler& sampler):
    sampler { sampler },
    parents {},
    depth { 0 },
    existingOps {}
{
    ++CountCreations;
    ++CountFinalizabilityMaybe;
}

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
    if (inConstruction()) {
        return Finalizability::Maybe;
    }
    return GetFinalizabilityFromState(state);
}

} // namespace kas
