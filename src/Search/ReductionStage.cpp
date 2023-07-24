#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

ReductionStage::Expander::Expander(std::size_t numWorkers) {
    for (std::size_t i = 0; i < numWorkers; ++i) {
        workers.emplace_back([this](std::stop_token stopToken) {
            while (!stopToken.stop_requested()) {
                ReductionStage *stage = nullptr;
                {
                    auto expanderLock = lock();
                    if (cv.wait(expanderLock, stopToken, [&] {
                        return !queue.empty();
                    })) {
                        stage = queue.front();
                        queue.pop();
                    }
                }
                if (stage) {
                    stage->expand(*this);
                    auto expanderLock = lock();
                    ++completed;
                    if (submitted == completed) {
                        expanderLock.unlock();
                        cvReady.notify_all();
                    }
                }
            }
        });
    }
}

void ReductionStage::Expander::add(std::unique_lock<std::mutex>&, ReductionStage *stage) {
    ++submitted;
    queue.emplace(stage);
    cv.notify_one();
}

void ReductionStage::Expander::addRoot(ReductionStage *stage) {
    auto expanderLock = lock();
    add(expanderLock, stage);
    cvReady.wait(expanderLock, [&] {
        return submitted == completed;
    });
}

ReductionStage::Expander::~Expander() {
    for (auto& worker: workers) {
        worker.request_stop();
    }
}

void ReductionStage::expand(Expander& expander) {
    Lock lock = acquireLock();
    if (expanded) {
        return;
    }

    const auto& options = sampler.getOptions();

    // First create the corresponding NormalStage.
    // Note: you cannot getNextOp, because it shares the same hash with us.
    // The lock for the NormalStage and this are the same, because we share the same hash and depth.
    // There is no dead lock because we use std::recursive_lock.
    std::tie(nStage, lock) = NormalStage::Create(getInterface(), *this, std::nullopt, std::move(lock));
    auto nStageFinalizability = nStage->getFinalizability(lock);

    if (nStageFinalizability == Finalizability::Yes) {
        // If this is true, then this stage is finalizable.
        // We do not even need to know the finalizability of the other children.
        determineFinalizability(Finalizability::Yes, true);
    }

    // Check if there is need to generate new stages.
    if (
        existingOp<MapReduceOp>() >= options.maximumReductions
        || existingOp<MapReduceOp>() >= options.depth
    ) {
        if (nStageFinalizability == Finalizability::No) {
            // This stage is dead.
            determineFinalizability(nStageFinalizability, true);
        }
        return;
    }

    // Then attempt to generate new reductions.
    std::vector<const MapReduceOp *> reductions;
    std::ranges::move(getInterface() | std::views::transform([](const Dimension& dim) { return dynamic_cast<const MapReduceOp *>(dim.tryAs<MapReduce>()); }) | std::views::filter([](auto ptr) { return ptr != nullptr; }), std::back_inserter(reductions));
    KAS_ASSERT(reductions.size() == existingOp<MapReduceOp>());
    std::ranges::sort(reductions, std::less<>{}, &MapReduceOp::getPriority);

    std::vector<NextStageSlot> nextReductions;
    for (auto op: MapReduceOp::Generate(sampler.getOpStore(), reductions, {
        .ctx = sampler.getBindingContext(),
        .dimUpperBound = options.dimUpperBound,
        .outputSize = sampler.getTotalOutputSize(),
        .maxFLOPs = options.maxFLOPs,
    })) {
        ReductionStage *stage;
        {
            Lock lock;
            std::tie(stage, lock) = getNextOp(op);
        }
        nextReductions.emplace_back(Next{Next::Type::MapReduce, NextStageSlot::GetKey(op)}, op, stage);
    }
    nextSlotStore.fill(nextReductions, [](NextStageSlot& slot) -> NextStageSlot&& {
        return std::move(slot);
    });
    nextSlotStore.checkHashCollisionAndRemove();
    expanded = true;

    if (nextSlotStore.size() == 0 && nStageFinalizability == Finalizability::No) {
        // Clearly we are dead.
        // Need to propagate because we are the last reduction stage.
        determineFinalizability(Finalizability::No, true);
        return;
    }

    auto expanderLock = expander.lock();
    nextSlotStore.forEach([&](const NextStageSlot& slot) {
        // Add children to queue.
        expander.add(expanderLock, static_cast<ReductionStage *>(slot.nextStage));
    });
}

ReductionStage::CollectedFinalizabilities ReductionStage::collectFinalizabilities() {
    return { Base::collectFinalizabilities(), nStage->getFinalizability() };
}

Finalizability ReductionStage::checkForFinalizableChildren(const CollectedFinalizabilities& collected) const {
    auto rStageFinalizability = Base::checkForFinalizableChildren(collected);
    return rStageFinalizability + collected.nStageFinalizability;
}

ReductionStage::ReductionStage(Sampler& sampler, Dimensions interface, Lock lock):
    Base { sampler, std::move(interface), std::move(lock) }
{}

ReductionStage::ReductionStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock):
    Base { std::move(interface), creator, std::move(deltaOp), std::move(lock) }
{}

std::size_t ReductionStage::countChildrenImpl() const {
    return Base::countChildrenImpl() + nStage->countChildren();
}

std::vector<Next> ReductionStage::getChildrenHandlesImpl() {
    std::vector<Next> handles = Base::getChildrenHandlesImpl();
    std::ranges::move(nStage->getChildrenHandles(), std::back_inserter(handles));
    return handles;
}

std::vector<Arc> ReductionStage::getChildrenArcsImpl() {
    std::vector<Arc> arcs = Base::getChildrenArcsImpl();
    std::ranges::move(nStage->getChildrenArcs(), std::back_inserter(arcs));
    return arcs;
}

std::optional<Arc> ReductionStage::getArcFromHandleImpl(Next next) {
    if (next.type == Next::Type::MapReduce) {
        return Base::getArcFromHandleImpl(next);
    } else {
        return nStage->getArcFromHandle(next);
    }
}

std::optional<Node> ReductionStage::getChildImpl(Next next) {
    if (next.type == Next::Type::MapReduce) {
        return Base::getChildImpl(next);
    } else {
        return nStage->getChild(next);
    }
}

bool ReductionStage::canAcceptArcImpl(Arc arc) {
    return arc.match<bool>(
        [&](const PrimitiveOp *op) -> bool {
            if (op->getType() == DimensionType::MapReduce) {
                // We have to manually find if this is in the search space.
                auto newInterface = op->applyToInterface(getInterface());
                Lock lock = Lock { sampler.getMutex(depth + 1, newInterface)};
                return sampler.getStageStore().find(depth + 1, newInterface, lock) != nullptr;
            } else {
                return nStage->canAcceptArc(arc);
            }
        },
        [&](auto) -> bool {
            return nStage->canAcceptArc(arc);
        }
    );
}

Node ReductionStage::getChildImpl(Arc arc) {
    return arc.match<Node>(
        [&](auto op) -> Node {
            if (op->getType() == DimensionType::MapReduce) {
                return { &sampler, getNextOpWithoutLock(op) };
            } else {
                return nStage->getChildImpl(arc);
            }
        },
        [&](auto op) -> Node {
            return nStage->getChild(arc);
        }
    );
}

} // namespace kas
