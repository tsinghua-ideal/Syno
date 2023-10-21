#include "KAS/Search/AbstractStage.hpp"


namespace kas {

AbstractStage::Lock AbstractStage::obtainInitialLock() {
    KAS_ASSERT(initialLock.owns_lock(), "Initial lock has already been obtained.");
    return std::move(initialLock);
}

bool AbstractStage::tryExpandingMoreLayers(int layers) {
    // If layers > expandedLayers, actually perform the expansion.
    // Otherwise, ignore.
    while (true) {
        int existingExpandedLayers = expandedLayers;
        if (layers <= existingExpandedLayers) {
            return false;
        }
        if (expandedLayers.compare_exchange_weak(existingExpandedLayers, layers)) {
            break;
        }
    }
    return true;
}

void AbstractStage::determineFinalizability(Finalizability yesOrNo, bool propagate) {
    KAS_ASSERT(!isFinalizabilityDetermined(), "Finalizability has already been determined.");
    switch (yesOrNo) {
    case Finalizability::Yes:
        --CountFinalizabilityMaybe;
        ++CountFinalizabilityYes;
        state = Finalizability::Yes;
        break;
    case Finalizability::No:
        --CountFinalizabilityMaybe;
        ++CountFinalizabilityNo;
        state = Finalizability::No;
        break;
    default:
        KAS_CRITICAL("Invalid Finalizability.");
    }
    if (propagate) {
        for (auto parent: parents) {
            parent->requestFinalizabilityUpdate(this);
        }
    }
}

void AbstractStage::requestFinalizabilityUpdate(const AbstractStage *requestor) {
    sampler.getPruner().requestFinalizabilityUpdate(this, requestor);
}

AbstractStage::AbstractStage(Sampler &sampler, GraphHandle interface, Lock lock):
    initialLock{ [&]() -> Lock {
        if (!lock.owns_lock()) {
            return Lock { sampler.getMutex(AbstractStage::GetRootMutexIndex(interface)) };
        } else {
            return std::move(lock);
        }
    }() },
    mutex { *initialLock.mutex() },
    parents {},
    sampler { sampler },
    interface(std::move(interface)),
    depth { 0 },
    existingOps{}
{
    ++CountCreations;
    ++CountFinalizabilityMaybe;
}

AbstractStage::AbstractStage(GraphHandle interface, AbstractStage& creator, std::optional<Next::Type> optionalDeltaOp, Lock lock):
    initialLock { [&]() -> Lock {
        if (!lock.owns_lock()) {
            return Lock { creator.sampler.getMutex(creator.getNextMutexIndex(optionalDeltaOp.has_value(), interface)) };
        } else {
            return std::move(lock);
        }
    }() },
    mutex { *initialLock.mutex() },
    parents { &creator },
    sampler { creator.sampler },
    interface(std::move(interface)),
    depth { creator.depth + static_cast<std::size_t>(optionalDeltaOp.has_value()) },
    existingOps { creator.existingOps }
{
    ++CountCreations;
    ++CountFinalizabilityMaybe;
    if (optionalDeltaOp) {
        Next::Type deltaOp = *optionalDeltaOp;
        existingOps[deltaOp] += 1;
        switch (deltaOp) {
        case Next::Type::Reduce: ++CountChildrenReduce; break;
        case Next::Type::Expand: ++CountChildrenExpand; break;
        case Next::Type::Shift: ++CountChildrenShift; break;
        case Next::Type::Stride: ++CountChildrenStride; break;
        case Next::Type::Split: ++CountChildrenSplit; break;
        case Next::Type::Unfold: ++CountChildrenUnfold; break;
        case Next::Type::Merge: ++CountChildrenMerge; break;
        case Next::Type::Share: ++CountChildrenShare; break;
        default: KAS_UNREACHABLE();
        }
    }
}

AbstractStage::Lock AbstractStage::addParent(AbstractStage &parent) {
    Lock lock = acquireLock();
    parents.emplace_back(&parent);
    return lock;
}

void AbstractStage::addParent(AbstractStage &parent, Lock &lock) {
    KAS_ASSERT(lock.owns_lock());
    parents.emplace_back(&parent);
}

std::size_t AbstractStage::remainingDepth() const {
    const std::size_t maxDepth = sampler.getOptions().depth;
    KAS_ASSERT(maxDepth >= depth);
    return maxDepth - depth;
}

std::size_t AbstractStage::hash() const {
    return std::hash<GraphHandle>{}(interface);
}
std::string AbstractStage::description() const {
    return interface.description(sampler.getBindingContext());
}

void AbstractStage::expand(int layers) {
    if (tryExpandingMoreLayers(layers)) {
        sampler.getExpander().expand(toNode(), layers);
    }
}
void AbstractStage::expandSync(int layers) {
    if (tryExpandingMoreLayers(layers)) {
        sampler.getExpander().expandSync(toNode(), layers);
    }
}

} // namespace kas
