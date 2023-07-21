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
                    std::unique_lock lock(mutex);
                    if (cv.wait(lock, stopToken, [&] {
                        return !queue.empty();
                    })) {
                        stage = queue.front();
                        queue.pop();
                    }
                }
                if (stage) {
                    stage->expand(*this);
                    ++completed;
                    if (submitted == completed) {
                        cvReady.notify_all();
                    }
                }
            }
        });
    }
}

void ReductionStage::Expander::add(ReductionStage *stage, Lock) {
    {
        std::scoped_lock lock(mutex);
        ++submitted;
        queue.emplace(stage);
    }
    cv.notify_one();
}

void ReductionStage::Expander::addRoot(ReductionStage *stage, Lock stageLock) {
    add(stage, std::move(stageLock));
    std::unique_lock lock { mutex };
    cvReady.wait(lock, [&] {
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
    Finalizability fin = nStage->getFinalizability(lock);

    // Then attempt to generate new reductions.
    if (
        existingOp<MapReduceOp>() >= options.maximumReductions
        || existingOp<MapReduceOp>() >= options.depth
    ) {
        if (fin != Finalizability::Maybe) {
            // No need to propagate because the reductions are built recursively.
            determineFinalizability(fin, false);
        }
        return;
    }

    std::vector<const MapReduceOp *> reductions;
    std::ranges::move(getInterface() | std::views::transform([](const Dimension& dim) { return dynamic_cast<const MapReduceOp *>(dim.tryAs<MapReduce>()); }) | std::views::filter([](auto ptr) { return ptr != nullptr; }), std::back_inserter(reductions));
    KAS_ASSERT(reductions.size() == existingOp<MapReduceOp>());
    std::ranges::sort(reductions, std::less<>{}, &MapReduceOp::getPriority);

    std::vector<NextStageSlot> nextReductions;
    std::map<AbstractStage *, Finalizability> childrenFinalizabilities;
    for (auto op: MapReduceOp::Generate(sampler.getOpStore(), reductions, {
        .ctx = sampler.getBindingContext(),
        .dimUpperBound = options.dimUpperBound,
        .outputSize = sampler.getTotalOutputSize(),
        .maxFLOPs = options.maxFLOPs,
    })) {
        ReductionStage *stage;
        Finalizability f;
        {
            Lock lock;
            std::tie(stage, lock) = getNextOp(op);
            f = stage->getFinalizability(lock);
            expander.add(stage, std::move(lock));
        }
        if (f != Finalizability::No) {
            nextReductions.emplace_back(Next{Next::Type::MapReduce, NextStageSlot::GetKey(op)}, op, stage);
            childrenFinalizabilities.emplace(stage, f);
        }
    }
    nextSlotStore.fill(nextReductions, [](NextStageSlot& slot) -> NextStageSlot&& {
        return std::move(slot);
    });
    nextSlotStore.checkHashCollisionAndRemove();
    nextSlotStore.forEach([&](const NextStageSlot& slot) {
        auto f = childrenFinalizabilities[slot.nextStage];
        fin += f;
    });
    if (fin != Finalizability::Maybe) {
        // No need to propagate because the reductions are built recursively.
        determineFinalizability(fin, false);
    }

    expanded = true;
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
