#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

void ReductionStage::expand() {
    const auto& options = sampler.getOptions();

    // First create the corresponding NormalStage.
    // Note: you cannot getNextOp, because it shares the same hash with us.
    Finalizability fin;
    {
        // Actually this lock is same as initialLock, because we share the same hash and depth.
        // This is OK because we use std::recursive_lock.
        Lock lock;
        std::tie(nStage, lock) = NormalStage::Create(getInterface(), *this, std::nullopt, Lock{});
        fin = nStage->getFinalizability(lock);
    }

    // Then attempt to generate new reductions.
    if (existingOp<MapReduceOp>() == options.maximumReductions) {
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
{
    expand();
}

ReductionStage::ReductionStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock):
    Base { std::move(interface), creator, std::move(deltaOp), std::move(lock) }
{
    expand();
}

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
    if(next.type == Next::Type::MapReduce) {
        return Base::getArcFromHandleImpl(next);
    } else {
        return nStage->getArcFromHandle(next);
    }
}

std::optional<Node> ReductionStage::getChildImpl(Next next) {
    if(next.type == Next::Type::MapReduce) {
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
