#include <unordered_map>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::size_t NormalStageStore::Hash::operator()(const Dimensions& interface) const noexcept {
    return std::hash<Dimensions>{}(interface);
}

std::size_t NormalStageStore::Hash::operator()(const NormalStage * nStage) const noexcept {
    return (*this)(nStage->getInterface());
}

bool NormalStageStore::Equal::operator()(const Dimensions& lhs, const Dimensions& rhs) const noexcept {
    return std::ranges::equal(lhs, rhs);
}
bool NormalStageStore::Equal::operator()(const Dimensions& lhs, const NormalStage *rhs) const noexcept {
    return (*this)(lhs, rhs->getInterface());
}
bool NormalStageStore::Equal::operator()(const NormalStage *lhs, const Dimensions& rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs);
}
bool NormalStageStore::Equal::operator()(const NormalStage *lhs, const NormalStage *rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs->getInterface());
}

NormalStage *NormalStageStore::find(const Dimensions& interface) const {
    KAS_ASSERT(std::ranges::is_sorted(interface, Dimension::HashLessThan{}), "Interface is not sorted.");
    if (auto it = interfaces.find(interface); it != interfaces.end()) {
        return *it;
    } else {
        return nullptr;
    }
}

bool NormalStageStore::insert(NormalStage *nStage) {
    return interfaces.insert(nStage).second;
}

NormalStageStore::~NormalStageStore() {
    for (auto interface: interfaces) {
        delete interface;
    }
}

NormalStageStore& NormalStage::getNormalStageStore() {
    return sampler.getNormalStageStore();
}

void NormalStage::removeDeadChildrenFromSlots() {
    KAS_ASSERT(childrenGenerated);
    nextOpStores.forEach([&](auto& store) {
        store.remove([&](const auto& slot) {
            // Even if the stage is removed, we had better keep it in memory so we avoid redundant computation.
            return slot.nextStage->getFinalizability() == Finalizability::No;
        });
    });
}

void NormalStage::removeAllChildrenFromSlots() {
    KAS_ASSERT(childrenGenerated);
    nextOpStores.forEach([&](auto& store) {
        store.clear();
    });
}

AbstractStage::Finalizability NormalStage::checkForFinalizableChildren() const {
    KAS_ASSERT(childrenGenerated);
    // If there are FinalizeOp's, of course this is finalizable.
    if (nextFinalizations.size()) {
        return Finalizability::Yes;
    }
    // Otherwise, check children. Yes if any Yes, No if all No.
    bool allNo = true;
    bool foundYes = false;
    nextOpStores.forEach([&](const auto& store) {
        store.forEach([&](const auto& slot) {
            if (foundYes) {
                return;
            }
            NormalStage *child = slot.nextStage;
            if (child->getFinalizability() == Finalizability::Yes) {
                foundYes = true;
                allNo = false;
            } else if (child->getFinalizability() == Finalizability::Maybe) {
                allNo = false;
            }
        });
    });
    if (foundYes) {
        return Finalizability::Yes;
    } else if (allNo) {
        return Finalizability::No;
    } else {
        return Finalizability::Maybe;
    }
}

void NormalStage::guardGeneratedChildren() {
    if (childrenGenerated) {
        return;
    }
    KAS_ASSERT(!generatingChildren);
    generatingChildren = true;
    const BindingContext& ctx = sampler.getBindingContext();
    const SampleOptions& options = sampler.getOptions();
    PrimitiveOpStore& store = sampler.getOpStore();
    Graph graph = interface.buildGraph();

    // Now we start to modify the states.
    auto guard = acquireFinalizabilityLock();

    // First add finalizations.
    nextFinalizations.fill(FinalizeOp::Generate(interface, graph, {
        .ctx = ctx,
        .desired = sampler.getInputShape(),
        .maximumTensors = options.maximumTensors,
    }), [](FinalizeOp& f) {
        return NextFinalizeSlot({NextFinalizeSlot::GetKey(f.tensors)}, std::move(f));
    });
    nextFinalizations.checkHashCollisionAndRemove();

    // Wrap the generated Op in a NextOpSlot.
    auto add = [&]<PrimitiveOpImpl Op>(const std::vector<const Op *>& newOps) {
        nextOpStores.get<Op>().fill(
            newOps,
            [&](const Op *op) {
                return NextOpSlot<Op>({NextOpSlot<Op>::GetKey(op)}, op, getNextOp(op));
            }
        );
        const auto& rawSlots = nextOpStores.get<Op>().getRawSlots();
        if (auto it = std::ranges::adjacent_find(rawSlots); it != rawSlots.end()) {
            KAS_REPORT_OP_HASH_COLLISION(*it->op, *std::next(it)->op);
            nextOpStores.get<Op>().checkHashCollisionAndRemove();
        }
    };

    if (remainingDepth() > 0) {
        // Keep dimensionality, by applying `RepeatLikeOp`^{-1}s.
        // Shift^{-1}, TODO
        // Stride^{-1}
        if (options.maximumStrides == -1 || options.maximumStrides > existingOp<StrideOp>()) {
            add(StrideOp::Generate(store, interface, {
                .ctx = ctx,
                .maxStridedDimSize = options.maxStridedDimSize,
                .disallowStrideAboveSplit = options.disallowStrideAboveSplit,
                .disallowStrideAboveMergeR = options.disallowStrideAboveMergeR,
            }));
        }

        // Try decreasing dimensionality, by applying `SplitLikeOp`^{-1}s.
        if (interface.size() > options.dimLowerBound) {
            // Split^{-1}
            if (options.maximumSplits == -1 || options.maximumSplits > existingOp<SplitOp>()) {
                add(SplitOp::Generate(store, interface, {
                    .disallowDiscontinuousView = options.disallowDiscontinuousView,
                    .disallowSplitRAboveUnfold = options.disallowSplitRAboveUnfold,
                    .disallowSplitRAboveStride = options.disallowSplitRAboveStride,
                }));
            }
            // Unfold^{-1}
            if (options.maximumUnfolds == -1 || options.maximumUnfolds > existingOp<UnfoldOp>()) {
                add(UnfoldOp::Generate(store, interface, {
                    .ctx = ctx,
                    .minimumRatio = options.minimumUnfoldRatio,
                    .maxUnfoldKernelSize = options.maxUnfoldKernelSize,
                    .disallowUnfoldLAboveSplit = options.disallowUnfoldLAboveSplit,
                    .canonicalizeUnfoldOrder = options.canonicalizeUnfoldOrder,
                    .disallowUnfoldLAboveShift = options.disallowUnfoldLAboveShift,
                    .disallowUnfoldLAboveMergeR = options.disallowUnfoldLAboveMergeR,
                }));
            }
        }

        // Try increasing dimensionality, by applying `MergeLikeOp`^{-1}s.
        if (interface.size() < options.dimUpperBound) {
            // Merge^{-1}
            if (options.maximumMerges == -1 || options.maximumMerges > existingOp<MergeOp>()) {
                add(MergeOp::Generate(store, interface, {
                    .ctx = ctx,
                    .minimumRatio = options.minimumMergeRatio,
                    .disallowMergeWithLargeBlockAboveStride = options.disallowMergeWithLargeBlockAboveStride,
                    .disallowMergeWithLargeBlockAboveUnfold = options.disallowMergeWithLargeBlockAboveUnfold,
                }));
            }
            // Share^{-1}
            if (options.maximumShares == -1 || options.maximumShares > existingOp<ShareOp>()) {
                add(ShareOp::Generate(store, interface, {
                    .ctx = ctx,
                    .maximumTensors = options.maximumTensors,
                }));
            }
        }
    }

    CountChildrenFinalize += nextFinalizations.size();

    generatingChildren = false;
    childrenGenerated = true;

    requestUpdateForFinalizability();
    guard.releaseAndPropagateChanges();
}

std::shared_ptr<TensorView> NormalStage::getFinalize(const FinalizeOp *op) const {
    if (!op) return nullptr;
    // TODO!!!
    return op->buildTensorView(sampler.getFixedDimensions(), sampler.getExpressionForTensorNum(op->tensors.size()));
}

NormalStage *NormalStage::getNextOp(const PrimitiveOp *op) {
    NormalStageStore& store = getNormalStageStore();
    auto newInterface = op->applyToInterface(interface);
    if (NormalStage *found = store.find(newInterface); found) {
        found->addParent(*this);
        return found;
    } else {
        auto tempStage = std::make_unique<NormalStage>(std::move(newInterface), *this, Next::TypeOf(op->getType()));
        if(store.insert(tempStage.get())) {
            return tempStage.release();
        } else {
            KAS_CRITICAL("NormalStageStore::insert() failed.");
        }
    }
}

bool NormalStage::possibleToFinalizeByExperimenting() const {
    ++CountFinalizabilityCheckInvocations;

    const SampleOptions& options = sampler.getOptions();
    const BindingContext& ctx = sampler.getBindingContext();

    std::vector<Dimension> onlyWeights, weightsExcluded;
    for (const Dimension& dim: interface) {
        if (dim.deduceOrigin() == Dimension::Origin::Weight) {
            onlyWeights.emplace_back(dim);
        } else {
            weightsExcluded.emplace_back(dim);
        }
    }

    if (!FinalizeOp::FitIntoWeights(onlyWeights, {
        .maximumTensors = options.maximumTensors,
    })) {
        ++CountTooManyWeights;
        return false;
    }

    auto remaining = [&](int maximum, Next::Type existingType) {
        int existing = existingOps[existingType];
        return maximum == -1 ? static_cast<int>(options.depth) : std::max(maximum - existing, 0);
    };
    const std::size_t distance = FinalizeOp::Distance(weightsExcluded, sampler.getInputShape(), {
        .ctx = ctx,
        .remainingMerges = remaining(options.maximumMerges, Next::Type::Merge),
        .remainingSplits = remaining(options.maximumSplits, Next::Type::Split),
        .remainingUnfolds = remaining(options.maximumUnfolds, Next::Type::Unfold),
    });
    if (distance > remainingDepth()) {
        ++CountShapeDeviatesTooMuch;
        return false;
    }

    return true;
}

std::size_t NormalStage::uncheckedCountChildren() const {
    return nextFinalizations.size() + nextOpStores.size();
}

std::vector<Next> NormalStage::uncheckedGetChildrenHandles() const {
    auto nextF = nextFinalizations.toNexts();
    auto nexts = nextOpStores.toNexts();
    nexts.insert(nexts.begin(), nextF.begin(), nextF.end());
    return nexts;
}

std::vector<Arc> NormalStage::uncheckedGetArcs() const {
    auto arcsF = nextFinalizations.toArcs(&sampler);
    auto arcs = nextOpStores.toArcs(&sampler);
    arcs.insert(arcs.begin(), arcsF.begin(), arcsF.end());
    return arcs;
}

const NextFinalizeSlot *NormalStage::getChildFinalizeSlot(std::size_t key) const {
    return nextFinalizations.getSlot(key);
}

NormalStage::NormalStage(Dimensions&& interface, AbstractStage& creator, std::optional<Next::Type> deltaOp):
    AbstractStage { creator, deltaOp },
    interface { std::move(interface) }
{
    // Perform experimental finalization, i.e., compute the ShapeComplexity of the interface.
    if (!possibleToFinalizeByExperimenting()) {
        // If proved to be not finalizable, no need to generate children.
        childrenGenerated = true;
        // We cannot propagte, because the parent is now building children.
        determineFinalizability(Finalizability::No);
    }
}

std::size_t NormalStage::countChildren() {
    return guarded([this] { return uncheckedCountChildren(); });
}

std::vector<Next> NormalStage::getChildrenHandles() {
    return guarded([this] { return uncheckedGetChildrenHandles(); });
}

std::vector<Arc> NormalStage::getChildrenArcs() {
    return guarded([this] { return uncheckedGetArcs(); });
}

std::optional<Arc> NormalStage::getArcFromHandle(Next next) {
    return guarded([&] { return matchNext<Arc>(next,
        [&](const auto& slot) -> Arc {
            return { &sampler, slot.op };
        },
        [&](const auto& slot) -> Arc {
            return { &sampler, &slot.finalization };
        }
    );});
}

std::optional<Node> NormalStage::getChild(Next next) {
    return guarded([&] { return matchNext<Node>(next,
        [&](const auto& slot) -> Node {
            return { &sampler, slot.nextStage };
        },
        [&](const auto& slot) -> Node {
            return { &sampler, getFinalize(&slot.finalization) };
        }
    );});
}

Node NormalStage::getChild(Arc arc) {
    return arc.match<Node>(
        [&](auto op) -> Node {
            return { &sampler, getNextOp(op) };
        },
        [&](auto op) -> Node {
            return { &sampler, getFinalize(op) };
        }
    );
}

std::string NormalStage::description() const {
    return DimensionArrayToString(interface, sampler.getBindingContext());
}

} // namespace kas
