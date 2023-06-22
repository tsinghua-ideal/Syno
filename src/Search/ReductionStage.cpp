#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

void ReductionStage::expand() {
    const auto& options = sampler.getOptions();

    auto guard = acquireFinalizabilityLock();

    // First create the corresponding NormalStage.
    nStage = std::make_unique<NormalStage>(toInterface(), *this, std::nullopt);

    // Then attempt to generate new reductions.
    if (reductions.size() == options.maximumReductions) {
        return;
    }
    auto nextReductions = MapReduceOp::Generate(sampler.getOpStore(), reductions, {
        .ctx = sampler.getBindingContext(),
        .dimUpperBound = options.dimUpperBound,
        .outputSize = sampler.getTotalOutputSize(),
        .maxFLOPs = options.maxFLOPs,
    });
    this->nextReductions.fill(nextReductions, [&](const MapReduceOp *op) -> NextOpSlot<MapReduceOp> {
        return {{NextOpSlot<MapReduceOp>::GetKey(op)}, op, make(*this, op)};
    });
    this->nextReductions.checkHashCollisionAndRemove();

    requestUpdateForFinalizability();
    guard.releaseAndPropagateChanges();
}

void ReductionStage::removeDeadChildrenFromSlots() {
    nextReductions.remove([&](const NextOpSlot<MapReduceOp>& slot) {
        return slot.nextStage->getFinalizability() == Finalizability::No;
    });
}

void ReductionStage::removeAllChildrenFromSlots() {
    KAS_ASSERT(nStage->getFinalizability() == Finalizability::No);
    nextReductions.clear();
}

AbstractStage::Finalizability ReductionStage::checkForFinalizableChildren() const {
    bool allNo = true;
    bool foundYes = false;
    auto acc = [&](Finalizability fin) {
        if (fin == Finalizability::Yes) {
            foundYes = true;
            allNo = false;
        } else if (fin == Finalizability::Maybe) {
            allNo = false;
        }
    };
    acc(nStage->getFinalizability());
    nextReductions.forEach([&](const NextOpSlot<MapReduceOp>& slot) {
        if (foundYes) {
            return;
        }
        acc(slot.nextStage->getFinalizability());
    });
    if (foundYes) {
        return Finalizability::Yes;
    } else if (allNo) {
        return Finalizability::No;
    } else {
        return Finalizability::Maybe;
    }
}

const NextOpSlot<MapReduceOp> *ReductionStage::getChildSlot(std::size_t key) const {
    return nextReductions.getSlot(key);
}

ReductionStage::ReductionStage(ReductionStage& current, const MapReduceOp *nextReduction):
    AbstractStage { current, Next::Type::MapReduce }
{
    reductions = current.reductions;
    reductions.emplace_back(nextReduction);
    expand();
}

ReductionStage::ReductionStage(Sampler& sampler):
    AbstractStage { sampler }
{
    expand();
}

Dimensions ReductionStage::toInterface() const {
    auto interface = sampler.getRootInterface();
    std::ranges::move(reductions | std::views::transform(&MapReduceOp::getInput), std::back_inserter(interface));
    std::ranges::sort(interface, Dimension::HashLessThan{});
    return interface;
}

std::size_t ReductionStage::hash() const {
    return nStage->hash();
}

std::size_t ReductionStage::countChildren() {
    return nextReductions.size() + nStage->countChildren();
}

std::vector<Next> ReductionStage::getChildrenHandles() {
    std::vector<Next> handles = nextReductions.toNexts();
    std::ranges::move(nStage->getChildrenHandles(), std::back_inserter(handles));
    return handles;
}

std::vector<Arc> ReductionStage::getArcs() {
    std::vector<Arc> arcs = nextReductions.toArcs();
    std::ranges::move(nStage->getArcs(), std::back_inserter(arcs));
    return arcs;
}

std::optional<Arc> ReductionStage::getArcFromHandle(Next next) {
    if(next.type == Next::Type::MapReduce) {
        auto slot = getChildSlot(next.key);
        if (!slot) {
            return std::nullopt;
        }
        return slot->nextStage->lastReduction();
    } else {
        return nStage->getArcFromHandle(next);
    }
}

std::optional<Node> ReductionStage::getChild(Next next) {
    if(next.type == Next::Type::MapReduce) {
        auto slot = getChildSlot(next.key);
        if (!slot) {
            return std::nullopt;
        }
        return Node { &sampler, getChildSlot(next.key)->nextStage };
    } else {
        return nStage->getChild(next);
    }
}

Node ReductionStage::getChild(Arc arc) {
    return arc.match<Node>(
        [&](auto op) -> Node {
            if (op->getType() == DimensionType::MapReduce) {
                auto slot = getChildSlot(op->opHash());
                if (!slot) {
                    KAS_CRITICAL("Invalid Arc: no child with key {}.", op->opHash());
                }
                return Node { &sampler, slot->nextStage };
            } else {
                return nStage->getChild(Arc(op));
            }
        },
        [&](auto) -> Node {
            KAS_CRITICAL("Invalid Arc: applied to a final node.");
        }
    );
}

std::string ReductionStage::getChildDescription(Arc arc) {
    const auto& ctx = sampler.getBindingContext();
    return arc.match<std::string>(
        [&](auto op) -> std::string {
            return op->description(ctx);
        },
        [&](auto op) -> std::string {
            return op->description(ctx);
        }
    );
}

std::string ReductionStage::description() const {
    return nStage->description();
}

} // namespace kas
