#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

void NormalStage::removeDeadChildrenFromSlots(const CollectedFinalizabilities& collected) {
    if (!childrenGenerated) {
        return;
    }
    Base::removeDeadChildrenFromSlots(collected);
}

void NormalStage::removeAllChildrenFromSlots() {
    KAS_ASSERT(childrenGenerated);
    Base::removeAllChildrenFromSlots();
}

Finalizability NormalStage::checkForFinalizableChildren(const CollectedFinalizabilities& collected) const {
    if (!childrenGenerated) {
        return Finalizability::Maybe;
    }
    // If there are FinalizeOp's, of course this is finalizable.
    if (nextFinalizations.size()) {
        return Finalizability::Yes;
    }
    // Otherwise, check children.
    return Base::checkForFinalizableChildren(collected);
}

GraphHandle NormalStage::removeTooLongChains(const Graph& graph, const GraphHandle& interface) const {
    std::vector<Dimension> result;
    std::ranges::remove_copy_if(interface.getDimensions(), std::back_inserter(result), [&](const Dimension& dim) {
        return graph.getHeight(dim) >= sampler.getOptions().maxChainLength;
    });
    return GraphHandle(std::move(result), interface.getExpansions());
}

Size NormalStage::getAllowanceUsage(const Graph& graph) const {
    struct Visitor: OpVisitor {
        const BindingContext& ctx;
        bool countCoefficientsInWeightsAsUsage;
        Size result;
        Visitor(const BindingContext& ctx, bool countCoefficientsInWeightsAsUsage = false):
            ctx { ctx },
            countCoefficientsInWeightsAsUsage { countCoefficientsInWeightsAsUsage },
            result { Size::Identity(ctx) }
        {}
        void visit(const ExpandOp& op) override {}
        void visit(const ReduceOp& op) override {}
        void visit(const MergeOp& op) override {
            // s^(a+b) <- s^a, s^b yields (|a|+|b|-|a+b|)/2
            result *= (op.getBlock() * op.getGroup() / op.output.size()).sqrt();
        }
        void visit(const ShareOp& op) override {
            if (countCoefficientsInWeightsAsUsage) {
                result *= op.output.size().getAllowanceUsage();
            } else {
                // We must consider primary vars because they cannot be put in weights too many times.
                result *= op.output.size().primaryPart().getAllowanceUsage();
            }
        }
        void visit(const ShiftOp& op) override {}
        void visit(const SplitOp& op) override {}
        void visit(const StrideOp& op) override {
            result *= op.getStride().getAllowanceUsage();
        }
        void visit(const UnfoldOp& op) override {}
    };
    Visitor v {
        sampler.getBindingContext(),
        sampler.getOptions().countCoefficientsInWeightsAsAllowanceUsage
    };
    for (const PrimitiveOp *op: graph.getOps()) {
        op->accept(v);
    }
    const auto& reductions = graph.getReduceIterators();
    if (!reductions.empty()) {
        v.result *= ReductionShapeView(reductions).getAllowanceUsage();
    }
    return v.result;
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

    // First add finalizations.
    nextFinalizations.fill(FinalizeOp::Generate(interface, graph, {
        .ctx = ctx,
        .desired = sampler.getInputShape(),
        .maximumTensors = options.maximumTensors,
        .maximumFinalizations = options.maximumFinalizations,
        .allowWeightPermutation = options.allowWeightPermutation,
        .finalStageBuilder = [this](const FinalizeOp& op) {
            return getFinalize(op);
        },
        .maxFLOPs = options.maxFLOPs,
    }), [](std::pair<FinalizeOp, std::unique_ptr<FinalStage>>& opAndStage) {
        auto& [op, stage] = opAndStage;
        auto key = NextFinalizeSlot::GetKey(op.tensors);
        return NextFinalizeSlot({Next::Type::Finalize, key}, std::move(op), std::move(stage));
    });
    nextFinalizations.checkHashCollisionAndRemove();

    std::vector<NextStageSlot> children;
    std::map<AbstractStage *, Finalizability> childrenFinalizabilities;
    // Wrap the generated Op in a NextStageSlot.
    auto add = [&]<PrimitiveOpImpl Op>(const std::vector<const Op *>& newOps) {
        children.reserve(children.size() + newOps.size());
        for (const Op *op: newOps) {
            AbstractStage *stage;
            Finalizability fin;
            {
                Lock lock;
                std::tie(stage, lock) = getNextOp(op);
                fin = stage->getFinalizability(lock);
            }
            if (fin != Finalizability::No) {
                children.emplace_back(Next::FromOp(op), op, stage);
                childrenFinalizabilities.emplace(stage, fin);
            }
        }
    };

    auto prospectiveInterface = removeTooLongChains(graph, interface);
    const auto allowance = Allowance { ctx, getAllowanceUsage(graph), options.countCoefficientsInWeightsAsAllowanceUsage };

    if (remainingDepth() > 0) {
        // Keep dimensionality, by applying `RepeatLikeOp`^{-1}s.
        // Shift^{-1}
        if (!inCriticalState && (options.maximumShifts == -1 || options.maximumShifts > existingOp<ShiftOp>())) {
            add(ShiftOp::Generate(store, prospectiveInterface, {
                .ctx = ctx,
                .disallowShiftAboveUnfold = options.disallowShiftAboveUnfold,
                .maximumValidReshapeShiftPattern = options.maximumValidReshapeShiftPattern,
            }));
        }
        // Stride^{-1}
        if (!inCriticalState && (options.maximumStrides == -1 || options.maximumStrides > existingOp<StrideOp>())) {
            add(StrideOp::Generate(store, prospectiveInterface, {
                .ctx = ctx,
                .allowance = allowance,
                .maxStridedDimSize = options.maxStridedDimSize,
                .disallowStrideAboveSplit = options.disallowStrideAboveSplit,
                .disallowStrideAboveMergeR = options.disallowStrideAboveMergeR,
            }));
        }

        // Try decreasing dimensionality, by applying `SplitLikeOp`^{-1}s or `ExpandOp`^{-1}s.
        if (interface.getDimensions().size() > options.dimLowerBound) {
            // Split^{-1}
            if (options.maximumSplits == -1 || options.maximumSplits > existingOp<SplitOp>()) {
                add(SplitOp::Generate(store, prospectiveInterface, {
                    .ctx = ctx,
                    .graph = graph,
                    .disallowSplitLAboveUnfold = options.disallowSplitLAboveUnfold,
                    .disallowSplitRAboveUnfold = options.disallowSplitRAboveUnfold,
                    .disallowSplitRAboveStride = options.disallowSplitRAboveStride,
                }));
            }
            // Unfold^{-1}
            if (options.maximumUnfolds == -1 || options.maximumUnfolds > existingOp<UnfoldOp>()) {
                add(UnfoldOp::Generate(store, prospectiveInterface, {
                    .ctx = ctx,
                    .minimumRatio = options.minimumUnfoldRatio,
                    .maxUnfoldKernelSize = options.maxUnfoldKernelSize,
                    .requiresOddKernelSizeInUnfold = options.requiresOddKernelSizeInUnfold,
                    .disallowUnfoldLAboveSplit = options.disallowUnfoldLAboveSplit,
                    .canonicalizeUnfoldOrder = options.canonicalizeUnfoldOrder,
                    .disallowUnfoldLAboveShift = options.disallowUnfoldLAboveShift,
                    .disallowUnfoldLAboveMergeR = options.disallowUnfoldLAboveMergeR,
                }));
            }
            // Expand^{-1}
            if (options.maximumExpands == -1 || options.maximumExpands > existingOp<ExpandOp>()) {
                add(ExpandOp::Generate(store, prospectiveInterface, {
                    .ctx = ctx,
                    .disallowMergeInputAndWeight = options.disallowMergeInputAndWeight,
                    .disallowTile = options.disallowTile,
                    .disallowShareWeights = options.disallowShareWeights,
                    .maxExpansionRepeatMultiplier = options.maxExpansionRepeatMultiplier,
                    .maxExpansionMergeMultiplier = options.maxExpansionMergeMultiplier,
                }));
            }
        }

        // Try increasing dimensionality, by applying `MergeLikeOp`^{-1}s.
        if (interface.getDimensions().size() < options.dimUpperBound) {
            // Merge^{-1}
            if (options.maximumMerges == -1 || options.maximumMerges > existingOp<MergeOp>()) {
                add(MergeOp::Generate(store, prospectiveInterface, {
                    .ctx = ctx,
                    .allowance = allowance,
                    .disallowMergeWithLargeBlockAboveStride = options.disallowMergeWithLargeBlockAboveStride,
                    .disallowMergeWithLargeBlockAboveUnfold = options.disallowMergeWithLargeBlockAboveUnfold,
                    .maximumValidReshapeShiftPattern = options.maximumValidReshapeShiftPattern,
                }));
            }
            // Share^{-1}
            if (!inCriticalState && (options.maximumShares == -1 || options.maximumShares > existingOp<ShareOp>())) {
                add(ShareOp::Generate(store, prospectiveInterface, {
                    .allowance = allowance,
                    .maximumTensors = options.maximumTensors,
                }));
            }
        }
    }

    nextSlotStore.fill(children, [](NextStageSlot& child) -> NextStageSlot&& { return std::move(child); });
    const auto& rawSlots = nextSlotStore.getRawSlots();
    if (auto it = std::ranges::adjacent_find(rawSlots); it != rawSlots.end()) {
        KAS_REPORT_OP_HASH_COLLISION(*it->op, *std::next(it)->op);
        nextSlotStore.checkHashCollisionAndRemove();
    }
    Finalizability fin = nextFinalizations.size() > 0 ? Finalizability::Yes : Finalizability::No;
    nextSlotStore.forEach([&](const NextStageSlot& slot) {
        auto f = childrenFinalizabilities[slot.nextStage];
        fin += f;
    });

    CountChildrenFinalize += nextFinalizations.size();

    generatingChildren = false;
    childrenGenerated = true;

    if (fin != Finalizability::Maybe) {
        determineFinalizability(fin, true);
    }
}

std::unique_ptr<FinalStage> NormalStage::getFinalize(const FinalizeOp& op) {
    return std::make_unique<FinalStage>(*this, op.buildTensorView(
        sampler.getFixedDimensions(),
        sampler.getExpressionForTensorNum(op.tensors.size()),
        sampler.getBindingContext()
    ));
}

bool NormalStage::possibleToFinalizeByExperimenting() const {
    ++CountFinalizabilityCheckInvocations;

    const SampleOptions& options = sampler.getOptions();
    const BindingContext& ctx = sampler.getBindingContext();

    std::vector<Dimension> onlyWeights, weightsExcluded;
    for (const Dimension& dim: interface.getDimensions()) {
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
        .remainingUnfoldsAndExpands = remaining(options.maximumUnfolds, Next::Type::Unfold) + remaining(options.maximumExpands, Next::Type::Expand),
        .overflow = remainingDepth(),
    });
    if (distance > remainingDepth()) {
        ++CountShapeDeviatesTooMuch;
        return false;
    } else if (distance == remainingDepth()) {
        // Save this information.
        inCriticalState = true;
    }

    return true;
}

std::size_t NormalStage::uncheckedCountChildren() const {
    return nextFinalizations.size() + Base::countChildrenImpl();
}

std::vector<Next> NormalStage::uncheckedGetChildrenHandles() const {
    auto nextF = nextFinalizations.toNexts();
    auto nexts = Base::getChildrenHandlesImpl();
    nexts.insert(nexts.begin(), nextF.begin(), nextF.end());
    return nexts;
}

std::vector<Arc> NormalStage::uncheckedGetChildrenArcs() const {
    auto arcsF = nextFinalizations.toArcs(&sampler);
    auto arcs = Base::getChildrenArcsImpl();
    arcs.insert(arcs.begin(), arcsF.begin(), arcsF.end());
    return arcs;
}

const NextFinalizeSlot *NormalStage::getChildFinalizeSlot(Next next) const {
    return nextFinalizations.getSlot(next);
}

NormalStage::NormalStage(GraphHandle interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock):
    Base { std::move(interface), creator, std::move(deltaOp), std::move(lock) }
{
    // Perform experimental finalization, i.e., compute the ShapeComplexity of the interface.
    if (!possibleToFinalizeByExperimenting()) {
        // If proved to be not finalizable, no need to generate children.
        childrenGenerated = true;
        // We cannot propagte, because the parent is now building children.
        // Luckily our parent will first check the finalizability of this.
        determineFinalizability(Finalizability::No, false);
    } else {
        auto it = std::ranges::adjacent_find(interface.getDimensions(), [](const Dimension& a, const Dimension& b) {
            return a.hash() == b.hash();
        });
        if (it != interface.getDimensions().end()) {
            KAS_REPORT_DIMENSION_HASH_COLLISION(*it, *std::next(it));
        }
    }
}

std::size_t NormalStage::countChildrenImpl() {
    return guarded([this] { return uncheckedCountChildren(); });
}

std::vector<Next> NormalStage::getChildrenHandlesImpl() {
    return guarded([this] { return uncheckedGetChildrenHandles(); });
}

std::vector<Arc> NormalStage::getChildrenArcsImpl() {
    return guarded([this] { return uncheckedGetChildrenArcs(); });
}

std::optional<Arc> NormalStage::getArcFromHandleImpl(Next next) {
    return guarded([&next, this] {
        if (next.type == Next::Type::Finalize) {
            return nextFinalizations.findTransform<Arc>(next, [this](const NextFinalizeSlot& slot) -> Arc {
                return { &sampler, &slot.finalization };
            });
        }
        return Base::getArcFromHandleImpl(next);
    });
}

std::optional<Node> NormalStage::getChildImpl(Next next) {
    return guarded([&next, this] {
        if (next.type == Next::Type::Finalize) {
            auto slot = nextFinalizations.getSlot(next);
            if (!slot) return std::optional<Node>();
            return std::optional<Node>(std::in_place, &sampler, slot->nextStage.get());
        }
        return Base::getChildImpl(next);
    });
}

bool NormalStage::canAcceptArcImpl(Arc arc) {
    if (auto ptr = arc.tryAs<PrimitiveOp>(); ptr && ptr->getType() == DimensionType::Reduce) {
        // We have left ReductionStage.
        return false;
    }
    return Base::canAcceptArcImpl(arc);
}

Node NormalStage::getChildImpl(Arc arc) {
    return arc.match<Node>(
        [&](auto op) -> Node {
            return { &sampler, getNextOpWithoutLock(op) };
        },
        [&](auto op) -> Node {
            KAS_ASSERT(childrenGenerated);
            auto key = NextFinalizeSlot::GetKey(op->tensors);
            auto next = Next { Next::Type::Finalize, key };
            return getChildImpl(next).value();
        }
    );
}

} // namespace kas
