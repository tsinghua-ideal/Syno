#include "KAS/Core/Reduce.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Hash.hpp"


namespace kas {

void ReductionStage::expand(ThreadPool<ReductionStage *>& expander) {
    Lock lock = acquireLock();
    if (expanded) {
        return;
    }
    getStats().expandNode();

    const auto& ctx = sampler.getBindingContext();
    const auto& options = sampler.getOptions();

    // First create the corresponding NormalStage.
    // Note: you cannot getNextOp, because it shares the same hash with us.
    // The lock for the NormalStage and this are the same, because we share the same hash and depth.
    // There is no dead lock because we use std::recursive_lock.
    std::tie(nStage, lock) = NormalStage::Create(getMutexIndex(), getInterface(), *this, std::nullopt, std::move(lock));
    auto fin = nStage->experimentFinalizability(lock);

    // Check if there is need to generate new stages.
    if (
        existingOp<ReduceOp>() >= options.maximumReductions
        || existingOp<ReduceOp>() >= options.depth
    ) {
        if (fin != Finalizability::Maybe) {
            // This stage is determined.
            determineFinalizability(fin, true);
        }
        expanded = true;
        return;
    }

    // Then attempt to generate new reductions.
    std::vector<const Reduce *> reductions;
    std::ranges::move(getInterface().getDimensions() | std::views::transform([](const Dimension& dim) { return dim.tryAs<Reduce>(); }) | std::views::filter([](auto ptr) { return ptr != nullptr; }), std::back_inserter(reductions));
    KAS_ASSERT(reductions.size() == existingOp<ReduceOp>());
    std::ranges::sort(reductions, [](const Reduce *lhs, const Reduce *rhs) {
        return Reduce::LexicographicalLessThan(*lhs, *rhs);
    });

    // Get Allowance.
    Size currentReductionsUsage = reductions.empty() ? Size::Identity(ctx) : ReductionShapeView(reductions).getAllowanceUsage();
    Allowance allowance { ctx, currentReductionsUsage, options.countCoefficientsInWeightsAsAllowanceUsage };

    std::vector<NextStageSlot> nextReductions;
    std::map<AbstractStage *, Finalizability> childrenFinalizabilities;
    for (auto op: ReduceOp::Generate(sampler.getOpStore(), reductions, {
        .ctx = sampler.getBindingContext(),
        .allowance = allowance,
        .outputSize = sampler.getTotalOutputSize(),
        .maxRDomSizeBase = sampler.getMaxRDomSize(),
        .maxRDomSizeMultiplier = options.maxRDomSizeMultiplier,
        .maximumReductions = options.maximumReductions,
    })) {
        ReductionStage *stage;
        Finalizability f;
        {
            Lock lock;
            std::tie(stage, lock) = getNextOp(op);
            KAS_ASSERT(stage != nullptr); // Because we expand later.
            f = stage->getFinalizability(lock);
        }
        if (f != Finalizability::No) {
            nextReductions.emplace_back(Next::FromOp(op), op, stage);
            childrenFinalizabilities.try_emplace(stage, f);
        }
    }
    fillSlots(nextSlotStore, nextReductions, [](NextStageSlot& slot) -> NextStageSlot&& {
        return std::move(slot);
    });
    nextSlotStore.checkHashCollisionAndRemove();
    nextSlotStore.forEach([&](const NextStageSlot& slot) {
        auto f = childrenFinalizabilities.at(slot.nextStage);
        fin += f;
    });
    expanded = true;

    if (fin != Finalizability::Maybe) {
        // Need to propagate because we are the last reduction stage.
        determineFinalizability(fin, true);
    }

    expander.addMultiple(
        nextSlotStore.getRawSlots()
        | std::views::transform([](const NextStageSlot& slot) {
            return static_cast<ReductionStage *>(slot.nextStage);
        })
    );
}

ReductionStage::CollectedFinalizabilities ReductionStage::collectFinalizabilities() {
    return { Base::collectFinalizabilities(), nStage->getFinalizability() };
}

Finalizability ReductionStage::checkForFinalizableChildren(const CollectedFinalizabilities& collected) const {
    auto rStageFinalizability = Base::checkForFinalizableChildren(collected);
    return rStageFinalizability + collected.nStageFinalizability;
}

ReductionStage::ReductionStage(Sampler& sampler, GraphHandle interface, Lock lock):
    Base { sampler, std::move(interface), std::move(lock) }
{}

ReductionStage::ReductionStage(GraphHandle interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock):
    Base { std::move(interface), creator, std::move(deltaOp), std::move(lock) }
{}

std::size_t ReductionStage::countChildrenImpl() const {
    KAS_ASSERT(expanded);
    return Base::countChildrenImpl() + nStage->countChildren();
}

std::vector<Next> ReductionStage::getChildrenHandlesImpl() {
    KAS_ASSERT(expanded);
    std::vector<Next> handles = Base::getChildrenHandlesImpl();
    std::ranges::move(nStage->getChildrenHandles(), std::back_inserter(handles));
    return handles;
}

std::vector<Arc> ReductionStage::getChildrenArcsImpl() {
    KAS_ASSERT(expanded);
    std::vector<Arc> arcs = Base::getChildrenArcsImpl();
    std::ranges::move(nStage->getChildrenArcs(), std::back_inserter(arcs));
    return arcs;
}

std::optional<Arc> ReductionStage::getArcFromHandleImpl(Next next) {
    KAS_ASSERT(expanded);
    if (next.type == Next::Type::Reduce) {
        return Base::getArcFromHandleImpl(next);
    } else {
        return nStage->getArcFromHandle(next);
    }
}

std::optional<Node> ReductionStage::getChildImpl(Next next) {
    KAS_ASSERT(expanded);
    if (next.type == Next::Type::Reduce) {
        return Base::getChildImpl(next);
    } else {
        return nStage->getChild(next);
    }
}

bool ReductionStage::canAcceptArcImpl(Arc arc) {
    KAS_ASSERT(expanded);
    return arc.match<bool>(
        [&](const Operation *op) -> bool {
            if (auto reduceOp = dynamic_cast<const ReduceOp *>(op); reduceOp) {
                // We have to manually find if this is in the search space.
                auto newInterface = reduceOp->appliedToInterface(getInterface());
                Lock lock = Lock { sampler.getMutex(getNextMutexIndex(true, newInterface)) };
                return sampler.getStageStore().find(depth + 1, newInterface, lock) != nullptr;
            } else {
                return nStage->canAcceptArc(arc);
            }
        },
        [&](const FinalizeOp *) -> bool {
            return nStage->canAcceptArc(arc);
        }
    );
}

} // namespace kas
