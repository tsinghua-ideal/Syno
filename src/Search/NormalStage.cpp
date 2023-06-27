#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

void NormalStage::removeDeadChildrenFromSlots() {
    if(!childrenGenerated) {
        return;
    }
    DimensionsStage::removeDeadChildrenFromSlots();
}

void NormalStage::removeAllChildrenFromSlots() {
    KAS_ASSERT(childrenGenerated);
    DimensionsStage::removeAllChildrenFromSlots();
}

AbstractStage::Finalizability NormalStage::checkForFinalizableChildren() const {
    if(!childrenGenerated) {
        return Finalizability::Maybe;
    }
    // If there are FinalizeOp's, of course this is finalizable.
    if (nextFinalizations.size()) {
        return Finalizability::Yes;
    }
    // Otherwise, check children.
    return DimensionsStage::checkForFinalizableChildren();
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

    construct([&] { // Now we start to modify the states.

    // First add finalizations.
    nextFinalizations.fill(FinalizeOp::Generate(interface, graph, {
        .ctx = ctx,
        .desired = sampler.getInputShape(),
        .maximumTensors = options.maximumTensors,
        .maximumFinalizations = options.maximumFinalizations,
        .allowWeightPermutation = options.allowWeightPermutation,
    }), [](FinalizeOp& f) {
        return NextFinalizeSlot({Next::Type::Finalize, NextFinalizeSlot::GetKey(f.tensors)}, std::move(f));
    });
    nextFinalizations.checkHashCollisionAndRemove();

    // Wrap the generated Op in a NextOpSlot.
    auto add = [&]<PrimitiveOpImpl Op>(const std::vector<const Op *>& newOps) {
        nextSlotStore.fill(
            newOps,
            [&](const Op *op) {
                return NextDimensionsStageSlot({Next::TypeOf<Op>(), NextDimensionsStageSlot::GetKey(op)}, op, getNextOp<NormalStage>(op));
            }
        );
        const auto& rawSlots = nextSlotStore.getRawSlots();
        if (auto it = std::ranges::adjacent_find(rawSlots); it != rawSlots.end()) {
            KAS_REPORT_OP_HASH_COLLISION(*it->op, *std::next(it)->op);
            nextSlotStore.checkHashCollisionAndRemove();
        }
    };

    if (remainingDepth() > 0) {
        // Keep dimensionality, by applying `RepeatLikeOp`^{-1}s.
        // Shift^{-1}
        if (options.maximumShifts == -1 || options.maximumShifts > existingOp<ShiftOp>()) {
            add(ShiftOp::Generate(store, interface, {
                .disallowShiftAboveUnfold = options.disallowShiftAboveUnfold,
            }));
        }
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

    }); // End construction.
}

std::shared_ptr<TensorView> NormalStage::getFinalize(const FinalizeOp *op) const {
    if (!op) return nullptr;
    return op->buildTensorView(sampler.getFixedDimensions(), sampler.getExpressionForTensorNum(op->tensors.size()));
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
    return nextFinalizations.size() + DimensionsStage::uncheckedCountChildren();
}

std::vector<Next> NormalStage::uncheckedGetChildrenHandles() const {
    auto nextF = nextFinalizations.toNexts();
    auto nexts = DimensionsStage::uncheckedGetChildrenHandles();
    nexts.insert(nexts.begin(), nextF.begin(), nextF.end());
    return nexts;
}

std::vector<Arc> NormalStage::uncheckedGetChildrenArcs() const {
    auto arcsF = nextFinalizations.toArcs(&sampler);
    auto arcs = DimensionsStage::uncheckedGetChildrenArcs();
    arcs.insert(arcs.begin(), arcsF.begin(), arcsF.end());
    return arcs;
}

const NextFinalizeSlot *NormalStage::getChildFinalizeSlot(Next next) const {
    return nextFinalizations.getSlot(next);
}

NormalStage::NormalStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp):
    DimensionsStage { std::move(interface), creator, std::move(deltaOp) }
{
    // Perform experimental finalization, i.e., compute the ShapeComplexity of the interface.
    if (!possibleToFinalizeByExperimenting()) {
        // If proved to be not finalizable, no need to generate children.
        childrenGenerated = true;
        // We cannot propagte, because the parent is now building children.
        determineFinalizability(Finalizability::No);
    } else {
        auto it = std::ranges::adjacent_find(interface, [](const Dimension& a, const Dimension& b) {
            return a.hash() == b.hash();
        });
        if (it != interface.end()) {
            KAS_REPORT_DIMENSION_HASH_COLLISION(*it, *std::next(it));
        }
    }
    finishInitialConstruction();
}

std::size_t NormalStage::countChildren() {
    return guarded([this] { return uncheckedCountChildren(); });
}

std::vector<Next> NormalStage::getChildrenHandles() {
    return guarded([this] { return uncheckedGetChildrenHandles(); });
}

std::vector<Arc> NormalStage::getChildrenArcs() {
    return guarded([this] { return uncheckedGetChildrenArcs(); });
}

std::optional<Arc> NormalStage::getArcFromHandle(Next next) {
    return guarded([&next, this] {
        if (next.type == Next::Type::Finalize) {
            return nextFinalizations.findTransform<Arc>(next, [this](const NextFinalizeSlot& slot) -> Arc {
                return { &sampler, &slot.finalization };
            });
        }
        return DimensionsStage::uncheckedGetArcFromHandle(next);
    });
}

std::optional<Node> NormalStage::getChild(Next next) {
    return guarded([&next, this] {
        if (next.type == Next::Type::Finalize) {
            return nextFinalizations.findTransform<Node>(next, [this](const NextFinalizeSlot& slot) -> Node {
                return { &sampler, getFinalize(&slot.finalization) };
            });
        }
        return DimensionsStage::uncheckedGetChild<NormalStage>(next);
    });
}

bool NormalStage::canAcceptArc(Arc arc) {
    if (auto ptr = arc.tryAs<PrimitiveOp>(); ptr && ptr->getType() == DimensionType::MapReduce) {
        // We have left ReductionStage.
        return false;
    }
    return DimensionsStage::canAcceptArc(arc);
}

Node NormalStage::getChild(Arc arc) {
    return arc.match<Node>(
        [&](auto op) -> Node {
            return { &sampler, getNextOp<NormalStage>(op) };
        },
        [&](auto op) -> Node {
            return { &sampler, getFinalize(op) };
        }
    );
}

} // namespace kas
