#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

void ReductionStage::expand() {
    const auto& options = sampler.getOptions();

    // First create the corresponding NormalStage.
    nStage = std::make_unique<NormalStage>(Dimensions(getInterface()), *this, std::nullopt);

    // Then attempt to generate new reductions.
    if (existingOp<MapReduceOp>() == options.maximumReductions) {
        finishInitialConstruction(); // Finish modifying states.
        return;
    }

    std::vector<const MapReduceOp *> reductions;
    std::ranges::move(getInterface() | std::views::transform([](const Dimension& dim) { return dynamic_cast<const MapReduceOp *>(dim.tryAs<MapReduce>()); }) | std::views::filter([](auto ptr) { return ptr != nullptr; }), std::back_inserter(reductions));
    KAS_ASSERT(reductions.size() == existingOp<MapReduceOp>());
    std::ranges::sort(reductions, std::less<>{}, &MapReduceOp::getPriority);

    auto nextReductions = MapReduceOp::Generate(sampler.getOpStore(), reductions, {
        .ctx = sampler.getBindingContext(),
        .dimUpperBound = options.dimUpperBound,
        .outputSize = sampler.getTotalOutputSize(),
        .maxFLOPs = options.maxFLOPs,
    });
    nextSlotStore.fill(nextReductions, [&](const MapReduceOp *op) -> NextDimensionsStageSlot {
        return {{Next::Type::MapReduce, NextDimensionsStageSlot::GetKey(op)}, op, getNextOp<ReductionStage>(op)};
    });
    nextSlotStore.checkHashCollisionAndRemove();

    finishInitialConstruction(); // Finish modifying states.
}

void ReductionStage::removeAllChildrenFromSlots() {
    KAS_ASSERT(nStage->getFinalizability() == Finalizability::No);
    DimensionsStage::removeAllChildrenFromSlots();
}

AbstractStage::Finalizability ReductionStage::checkForFinalizableChildren() const {
    auto nStageFinalizability = nStage->getFinalizability();
    auto rStageFinalizability = DimensionsStage::checkForFinalizableChildren();
    if (nStageFinalizability == Finalizability::Yes || rStageFinalizability == Finalizability::Yes) {
        return Finalizability::Yes;
    } else if (nStageFinalizability == Finalizability::No && rStageFinalizability == Finalizability::No) {
        return Finalizability::No;
    } else {
        return Finalizability::Maybe;
    }
}

ReductionStage::ReductionStage(Sampler& sampler):
    DimensionsStage { sampler, sampler.getRootInterface() }
{
    expand();
}

ReductionStage::ReductionStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp):
    DimensionsStage { std::move(interface), creator, std::move(deltaOp) }
{
    expand();
}

std::size_t ReductionStage::countChildren() {
    return uncheckedCountChildren() + nStage->countChildren();
}

std::vector<Next> ReductionStage::getChildrenHandles() {
    std::vector<Next> handles = uncheckedGetChildrenHandles();
    std::ranges::move(nStage->getChildrenHandles(), std::back_inserter(handles));
    return handles;
}

std::vector<Arc> ReductionStage::getChildrenArcs() {
    std::vector<Arc> arcs = uncheckedGetChildrenArcs();
    std::ranges::move(nStage->getChildrenArcs(), std::back_inserter(arcs));
    return arcs;
}

std::optional<Arc> ReductionStage::getArcFromHandle(Next next) {
    if(next.type == Next::Type::MapReduce) {
        return uncheckedGetArcFromHandle(next);
    } else {
        return nStage->getArcFromHandle(next);
    }
}

std::optional<Node> ReductionStage::getChild(Next next) {
    if(next.type == Next::Type::MapReduce) {
        return uncheckedGetChild<ReductionStage>(next);
    } else {
        return nStage->getChild(next);
    }
}

bool ReductionStage::canAcceptArc(Arc arc) {
    return arc.match<bool>(
        [&](const PrimitiveOp *op) -> bool {
            if (op->getType() == DimensionType::MapReduce) {
                // We have to manually find if this is in the search space.
                auto newInterface = op->applyToInterface(getInterface());
                return getStageStore().find(newInterface) != nullptr;
            } else {
                return nStage->canAcceptArc(arc);
            }
        },
        [&](auto) -> bool {
            return nStage->canAcceptArc(arc);
        }
    );
}

Node ReductionStage::getChild(Arc arc) {
    return arc.match<Node>(
        [&](auto op) -> Node {
            if (op->getType() == DimensionType::MapReduce) {
                return { &sampler, getNextOp<ReductionStage>(op) };
            } else {
                return nStage->getChild(arc);
            }
        },
        [&](auto op) -> Node {
            return nStage->getChild(arc);
        }
    );
}

} // namespace kas
