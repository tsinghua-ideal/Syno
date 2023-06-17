#include <unordered_map>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::size_t NormalStageStore::Hash::operator()(const ColoredInterface& interface) const noexcept {
    return std::hash<Interface>{}(interface.toDimensions());
}

std::size_t NormalStageStore::Hash::operator()(const NormalStage * nStage) const noexcept {
    return (*this)(nStage->getInterface());
}

bool NormalStageStore::Equal::operator()(const ColoredInterface& lhs, const ColoredInterface& rhs) const noexcept {
    return std::ranges::equal(lhs.toDimensions(), rhs.toDimensions());
}
bool NormalStageStore::Equal::operator()(const ColoredInterface& lhs, const NormalStage *rhs) const noexcept {
    return (*this)(lhs, rhs->getInterface());
}
bool NormalStageStore::Equal::operator()(const NormalStage *lhs, const ColoredInterface& rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs);
}
bool NormalStageStore::Equal::operator()(const NormalStage *lhs, const NormalStage *rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs->getInterface());
}

NormalStage *NormalStageStore::find(const ColoredInterface& interface) const {
    KAS_ASSERT(std::ranges::is_sorted(interface.toDimensions(), Dimension::HashLessThan{}), "Interface is not sorted.");
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

void NormalStage::determineFinalizability(Finalizability yesOrNo) {
    KAS_ASSERT(finalizability == Finalizability::Maybe, "Finalizability is already determined.");
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
}

void NormalStage::updateFinalizability() {
    // It would also be nice to remove all the dead-ends (Finalizability::No).
    auto removeDeadEnds = [&]() {
        // TODO: After API overhaul, remove dead-end children!
        // nextOpStores.forEach([&](auto& store) {
        //     store.remove([&](const auto& slot) {
        //         return slot.nextStage->finalizability == Finalizability::No;
        //     });
        // });
    };
    auto removeAllStageChildren = [&]() {
        nextOpStores.forEach([&](auto& store) {
            store.clear();
        });
    };

    if (finalizability == Finalizability::Yes) {
        removeDeadEnds();
        return;
    } else if (finalizability == Finalizability::No) {
        removeAllStageChildren();
        return;
    }
    // We need to check if this is finalizable.
    // If there are FinalizeOp, of course this is.
    if (nextFinalizations.size()) {
        determineFinalizability(Finalizability::Yes);
        removeDeadEnds();
        return;
    }
    // Otherwise, check children. Yes if any Yes, No if all No.
    bool allNo = true;
    bool foundYes = false;
    nextOpStores.forEach([&](auto& store) {
        store.forEach([&](const auto& slot) {
            if (foundYes) {
                return;
            }
            NormalStage *child = slot.nextStage;
            if (child->finalizability == Finalizability::Yes) {
                foundYes = true;
                allNo = false;
            } else if (child->finalizability == Finalizability::Maybe) {
                allNo = false;
            }
        });
    });
    if (foundYes) {
        determineFinalizability(Finalizability::Yes);
        removeDeadEnds();
        return;
    } else if (allNo) {
        determineFinalizability(Finalizability::No);
        removeAllStageChildren();
        return;
    }
    removeDeadEnds();
    return;
}

std::size_t NormalStage::remainingDepth() const {
    return sampler.getOptions().depth - depth;
}

void NormalStage::guard() {
    if (childrenGenerated) {
        updateFinalizability();
        return;
    }
    const BindingContext& ctx = sampler.getBindingContext();
    const SampleOptions& options = sampler.getOptions();
    DimensionStore& store = sampler.getDimStore();
    Graph graph = interface.buildGraph();

    // First add finalizations.
    nextFinalizations.fill(FinalizeOp::Generate(interface, graph, {
        .ctx = ctx,
        .desired = sampler.getInputShape(),
        .maximumTensors = options.maximumTensors,
    }), [](FinalizeOp& f) {
        return NextFinalizeSlot({NextFinalizeSlot::GetKey(f.tensors)}, std::move(f));
    });

    // Wrap the generated Op in a NextOpSlot.
    auto add = [&]<typename Op>(const std::vector<const Op *>& newOps) {
        nextOpStores.get<Op>()
            .fill(
                newOps,
                [&](const Op *op) {
                    return NextOpSlot<Op>({NextOpSlot<Op>::GetKey(op)}, op, getNextOp<Op>(op));
                }
            )
            .remove([&](const NextOpSlot<Op>& slot) {
                /* We need to call removeOp for deleted Op's and removeStage for deleted NormalStage's. TODO */
                return slot.nextStage->finalizability == Finalizability::No;
            });
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

    ++CountCreations;
    CountChildrenFinalize += nextFinalizations.size();
    CountChildrenShift += nextOpStores.get<ShiftOp>().size();
    CountChildrenStride += nextOpStores.get<StrideOp>().size();
    CountChildrenSplit += nextOpStores.get<SplitOp>().size();
    CountChildrenUnfold += nextOpStores.get<UnfoldOp>().size();
    CountChildrenMerge += nextOpStores.get<MergeOp>().size();
    CountChildrenShare += nextOpStores.get<ShareOp>().size();

    childrenGenerated = true;
    updateFinalizability();
}

std::shared_ptr<TensorView> NormalStage::getFinalize(std::size_t key) const {
    const auto& slot = uncheckedGetChildFinalizeSlot(key);
    KAS_DEBUG("Building TensorView from Finalization.");
    return slot.finalization.buildTensorView(sampler.getFixedDimensions());
}

bool NormalStage::possibleToFinalizeByExperimenting() const {
    ++CountFinalizabilityCheckInvocations;

    const SampleOptions& options = sampler.getOptions();
    const BindingContext& ctx = sampler.getBindingContext();

    std::vector<std::reference_wrapper<const ColoredDimension>> onlyWeights, weightsExcluded;
    for (const ColoredDimension& cDim : interface) {
        if (cDim.deduceOrigin() == ColoredDimension::Origin::Weight) {
            onlyWeights.emplace_back(std::cref(cDim));
        } else {
            weightsExcluded.emplace_back(std::cref(cDim));
        }
    }

    if (!FinalizeOp::FitIntoWeights(onlyWeights, {
        .maximumTensors = options.maximumTensors,
    })) {
        ++CountTooManyWeights;
        return false;
    }

    auto remaining = [&](int maximum, Next::Type existingType) {
        int existing = existingOps[static_cast<std::size_t>(existingType)];
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

const NextFinalizeSlot& NormalStage::uncheckedGetChildFinalizeSlot(std::size_t key) const {
    return nextFinalizations.getSlot(key);
}

Node NormalStage::uncheckedGetChild(Next next) const {
    switch (next.type) {
    case Next::Type::Shift: return { &sampler, uncheckedGetChildSlot<ShiftOp>(next.key).nextStage };
    case Next::Type::Stride: return { &sampler, uncheckedGetChildSlot<StrideOp>(next.key).nextStage };
    case Next::Type::Split: return { &sampler, uncheckedGetChildSlot<SplitOp>(next.key).nextStage };
    case Next::Type::Unfold: return { &sampler, uncheckedGetChildSlot<UnfoldOp>(next.key).nextStage };
    case Next::Type::Merge: return { &sampler, uncheckedGetChildSlot<MergeOp>(next.key).nextStage };
    case Next::Type::Share: return { &sampler, uncheckedGetChildSlot<ShareOp>(next.key).nextStage };
    case Next::Type::Finalize: return { &sampler, getFinalize(next.key) };
    default: KAS_UNREACHABLE("Invalid Next {}.", next.toString());
    }
}

NormalStage::NormalStage(Sampler& sampler, const std::vector<const MapReduceOp *>& reductions):
    interface { [&]() {
        auto interface = sampler.getRootInterface();
        std::ranges::copy(reductions, std::back_inserter(interface));
        std::ranges::sort(interface, Dimension::HashLessThan{});
        return interface;
    }() },
    sampler { sampler },
    depth { reductions.size() }
{
    existingOps[static_cast<std::size_t>(Next::Type::MapReduce)] = reductions.size();
    ++CountFinalizabilityMaybe;
    if (!possibleToFinalizeByExperimenting()) {
        determineFinalizability(Finalizability::No);
    }
}

NormalStage::NormalStage(Sampler& sampler, ColoredInterface&& interface, const NormalStage& old, Next::Type delta):
    interface { std::move(interface) },
    sampler { sampler },
    depth { old.depth + 1 },
    existingOps { old.existingOps }
{
    existingOps[static_cast<std::size_t>(delta)] += 1;
    ++CountFinalizabilityMaybe;
    if (!possibleToFinalizeByExperimenting()) {
        determineFinalizability(Finalizability::No);
    }
}

std::size_t NormalStage::countChildren() {
    guard();
    return uncheckedCountChildren();
}

std::vector<Next> NormalStage::getChildrenHandles() {
    guard();
    return uncheckedGetChildrenHandles();
}

const NextFinalizeSlot& NormalStage::getChildFinalizeSlot(std::size_t key) {
    guard();
    return uncheckedGetChildFinalizeSlot(key);
}

Node NormalStage::getChild(Next next) {
    guard();
    return uncheckedGetChild(next);
}

std::string NormalStage::description() const {
    return DimensionArrayToString(interface.toDimensions(), sampler.getBindingContext());
}

} // namespace kas
