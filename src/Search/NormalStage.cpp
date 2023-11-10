#include "KAS/Core/BindingContext.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

NormalStage::CollectedFinalizabilities NormalStage::collectFinalizabilities() {
    return {
        Base::collectFinalizabilities(),
        Base::collectFinalizabilities(nextContractions),
    };
}

void NormalStage::removeDeadChildrenFromSlots(const CollectedFinalizabilities& collected) {
    if (!childrenGenerated) {
        return;
    }
    Base::removeDeadChildrenFromSlots(collected);
    Base::removeDeadChildrenFromSlots(nextContractions, collected.contractionStageFinalizabilities);
}

void NormalStage::removeAllChildrenFromSlots() {
    KAS_ASSERT(childrenGenerated);
    Base::removeAllChildrenFromSlots();
    Base::removeAllChildrenFromSlots(nextContractions);
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
    return Base::checkForFinalizableChildren(collected) + Base::checkForFinalizableChildren(collected.contractionStageFinalizabilities);
}

void NormalStage::removeTooLongChains(ContractionOp::Analysis& analysis, const Graph& graph) const {
    auto tooLongChain = [&](const Dimension& dim) {
        return sampler.remainingChainLength(graph, dim) <= 0;
    };
    std::erase_if(analysis.simpleViewSearchable.getDimensions(), tooLongChain);
    std::erase_if(analysis.other, tooLongChain);
    std::erase_if(analysis.full.getDimensions(), tooLongChain);
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
    getStats().expandNode();

    KAS_ASSERT(shapeDistance.steps <= remainingDepth(), "You must first call possibleToFinalizeByExperimenting, then determineFinalizability before calling guardGeneratedChildren.");
    const bool inCriticalState = shapeDistance.steps == remainingDepth();

    const BindingContext& ctx = sampler.getBindingContext();
    const SampleOptions& options = sampler.getOptions();
    PrimitiveOpStore& store = sampler.getOpStore();
    ContractionOpStore& contractionStore = sampler.getContractionOpStore();
    Graph graph = interface.buildGraph();

    // First add finalizations.
    fillSlots(nextFinalizations, FinalizeOp::Generate(interface, graph, {
        .ctx = ctx,
        .desired = sampler.getDesiredShape(),
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
    }, true);
    nextFinalizations.checkHashCollisionAndRemove();

    std::vector<NextStageSlot> childrenView;
    std::vector<NextContractionSlot> childrenContraction;
    // Wrap the generated Op in a NextStageSlot or NextContractionSlot.
    auto add = [&]<GeneralizedOp Op>(auto& children, const std::vector<const Op *>& newOps) {
        children.reserve(children.size() + newOps.size());
        for (const Op *op: newOps) {
            NormalStage *stage;
            {
                Lock lock;
                std::tie(stage, lock) = getNextOp(op);
                if (stage != nullptr) {
                    KAS_ASSERT(stage->getFinalizability(lock) == Finalizability::Maybe);
                }
            }
            if (stage) {
                children.emplace_back(Next::FromOp(op), op, stage);
            }
        }
    };

    auto contractionAnalysis = ContractionOp::Analyze(interface, graph);
    removeTooLongChains(contractionAnalysis, graph);
    const auto& prospectiveInterface = contractionAnalysis.simpleViewSearchable;
    const auto allowance = Allowance { ctx, getAllowanceUsage(graph), options.countCoefficientsInWeightsAsAllowanceUsage };

    if (remainingDepth() > 0) {
        // Contraction^{-1}
        if (!inCriticalState && (options.maximumShares == -1 || options.maximumShares > contractionAnalysis.numShares)) {
            add(childrenContraction, ContractionOp::Generate(
                store, contractionStore, {
                    .analysis = contractionAnalysis,
                    .graph = graph,
                    .allowance = allowance,
                    .maximumTensors = options.maximumTensors,
                    .maxShares = options.maximumShares == -1 ? std::numeric_limits<int>::max() : options.maximumShares - contractionAnalysis.numShares,
                }
            ));
        }

        // Keep dimensionality, by applying `RepeatLikeOp`^{-1}s.
        // Shift^{-1}
        if (!inCriticalState && (options.maximumShifts == -1 || options.maximumShifts > existingOp<ShiftOp>())) {
            add(childrenView, ShiftOp::Generate(store, prospectiveInterface, {
                .ctx = ctx,
                .graph = graph,
                .disallowShiftAboveUnfold = options.disallowShiftAboveUnfold,
                .maximumValidReshapeShiftPattern = options.maximumValidReshapeShiftPattern,
            }));
        }
        // Stride^{-1}
        if (!inCriticalState && (options.maximumStrides == -1 || options.maximumStrides > existingOp<StrideOp>())) {
            add(childrenView, StrideOp::Generate(store, prospectiveInterface, {
                .ctx = ctx,
                .allowance = allowance,
                .maxStridedDimSize = options.maxStridedDimSize,
                .disallowStrideAboveSplit = options.disallowStrideAboveSplit,
                .disallowStrideAboveMergeR = options.disallowStrideAboveMergeR,
            }));
        }

        // Try decreasing dimensionality, by applying `SplitLikeOp`^{-1}s or `ExpandOp`^{-1}s.
        // Split^{-1}
        if (options.maximumSplits == -1 || options.maximumSplits > existingOp<SplitOp>()) {
            add(childrenView, SplitOp::Generate(store, contractionAnalysis.full, {
                .ctx = ctx,
                .graph = graph,
                .couldHaveBeenDoneBeforeLastContractionStage = contractionAnalysis.other,
                .disallowSplitLAboveUnfold = options.disallowSplitLAboveUnfold,
                .disallowSplitRAboveUnfold = options.disallowSplitRAboveUnfold,
                .disallowSplitRAboveStride = options.disallowSplitRAboveStride,
            }));
        }

        // Unfold^{-1}
        if (options.maximumUnfolds == -1 || options.maximumUnfolds > existingOp<UnfoldOp>()) {
            add(childrenView, UnfoldOp::Generate(store, contractionAnalysis.full, {
                .ctx = ctx,
                .graph = graph,
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
            add(childrenView, ExpandOp::Generate(store, prospectiveInterface, {
                .ctx = ctx,
                .disallowMergeInputAndWeight = options.disallowMergeInputAndWeight,
                .disallowTile = options.disallowTile,
                .disallowShareWeights = options.disallowShareWeights,
                .maxExpansionRepeatMultiplier = options.maxExpansionRepeatMultiplier,
                .maxExpansionMergeMultiplier = options.maxExpansionMergeMultiplier,
                .maxExpansionWeightsSharingDimSize = options.maxExpansionWeightsSharingDimSize,
                .minExpansionWeightsSharingDimSize = options.minExpansionWeightsSharingDimSize,
            }));
        }

        // Try increasing dimensionality, by applying `MergeLikeOp`^{-1}s.
        // Merge^{-1}
        if (options.maximumMerges == -1 || options.maximumMerges > existingOp<MergeOp>()) {
            add(childrenView, MergeOp::Generate(store, prospectiveInterface, {
                .ctx = ctx,
                .graph = graph,
                .allowance = allowance,
                .disallowMergeWithLargeBlockAboveStride = options.disallowMergeWithLargeBlockAboveStride,
                .disallowMergeWithLargeBlockAboveUnfold = options.disallowMergeWithLargeBlockAboveUnfold,
                .maximumValidReshapeShiftPattern = options.maximumValidReshapeShiftPattern,
            }));
        }
    }

    Finalizability fin = nextFinalizations.size() > 0 ? Finalizability::Yes : Finalizability::No;
    auto fill = [&]<typename Slot>(GenericNextSlotStore<Slot>& slotStore, std::vector<Slot>& children) {
        fillSlots(slotStore, children, [](Slot& child) -> Slot&& { return std::move(child); });
        const auto& rawSlots = slotStore.getRawSlots();
        if (auto it = std::ranges::adjacent_find(rawSlots); it != rawSlots.end()) {
            KAS_REPORT_OP_HASH_COLLISION(*it->op, *std::next(it)->op);
            slotStore.checkHashCollisionAndRemove();
        }
        if (slotStore.size()) {
            // Since we already know that all of the finalizabilities are Maybe.
            fin += Finalizability::Maybe;
        }
    };

    fill(nextContractions, childrenContraction);
    fill(nextSlotStore, childrenView);

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
    const Graph graph = interface.buildGraph();

    const std::size_t remainingSteps = remainingDepth();

    std::vector<CurrentDimension> current;
    std::vector<ColoredDimension> weightDims;
    for (const Dimension& dim: interface.getDimensions()) {
        const auto origin = dim.deduceOrigin(graph);
        int remainingLength = sampler.remainingChainLength(graph, dim);
        KAS_ASSERT(remainingLength >= 0);
        if (origin != Dimension::Origin::Weight) {
            if (origin == Dimension::Origin::UnfoldOrExpand && remainingLength <= 0) {
                ++CountShapeDeviatesTooMuch;
                return false;
            }
            current.emplace_back(dim, remainingLength);
        } else {
            weightDims.emplace_back(dim, dim.as<ShareOp::Input>().getRhsOrigin());
        }
    }

    auto remaining = [&](int maximum, Next::Type existingType) {
        int existing = existingOps[existingType];
        return maximum == -1 ? static_cast<int>(options.depth) : std::max(maximum - existing, 0);
    };
    const ShapeDistance distance = FinalizeOp::Distance(
        current, sampler.getDesiredShape(), graph,
        {
            .ctx = ctx,
            .remainingMerges = remaining(options.maximumMerges, Next::Type::Merge),
            .remainingSplits = remaining(options.maximumSplits, Next::Type::Split),
            .remainingUnfoldsAndExpands = remaining(options.maximumUnfolds, Next::Type::Unfold) + remaining(options.maximumExpands, Next::Type::Expand),
            .overflow = remainingSteps,
        },
        {
            .prune = options.enableFLOPsBasedPruning,
            .maximumTensors = options.maximumTensors,
            .maxFLOPs = options.maxFLOPs,
            .totalInputSize = sampler.getTotalInputSize(),
            .weightDims = weightDims,
        }
    );
    if (distance.steps > remainingSteps) {
        ++CountShapeDeviatesTooMuch;
        return false;
    }
    // Save this information.
    shapeDistance = distance;

    return true;
}

std::size_t NormalStage::uncheckedCountChildren() const {
    return nextFinalizations.size() + nextContractions.size() + Base::countChildrenImpl();
}

std::vector<Next> NormalStage::uncheckedGetChildrenHandles() const {
    auto nextF = nextFinalizations.toNexts();
    auto nextC = nextContractions.toNexts();
    auto nexts = Base::getChildrenHandlesImpl();
    nexts.insert(nexts.begin(), nextF.begin(), nextF.end());
    nexts.insert(nexts.begin(), nextC.begin(), nextC.end());
    return nexts;
}

std::vector<Arc> NormalStage::uncheckedGetChildrenArcs() const {
    auto arcsF = nextFinalizations.toArcs(&sampler);
    auto arcsC = nextContractions.toArcs(&sampler);
    auto arcs = Base::getChildrenArcsImpl();
    arcs.insert(arcs.begin(), arcsF.begin(), arcsF.end());
    arcs.insert(arcs.begin(), arcsC.begin(), arcsC.end());
    return arcs;
}

const NextFinalizeSlot *NormalStage::getChildFinalizeSlot(Next next) const {
    return nextFinalizations.getSlot(next);
}

NormalStage::NormalStage(GraphHandle interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock):
    Base { std::move(interface), creator, deltaOp, std::move(lock) }
{
    auto it = std::ranges::adjacent_find(interface.getDimensions(), [](const Dimension& a, const Dimension& b) {
        return a.hash() == b.hash();
    });
    if (it != interface.getDimensions().end()) {
        KAS_REPORT_DIMENSION_HASH_COLLISION(*it, *std::next(it));
    }
    if (!deltaOp.has_value()) {
        origin = NodeType::Reducing;
    } else if (deltaOp == Next::Type::Contraction) {
        origin = NodeType::Contraction;
    } else {
        origin = NodeType::Growing;
    }
}

AbstractStage *NormalStage::arbitraryParentImpl() const {
    if (origin == NodeType::Contraction) {
        // We are from ContractionStage. There should be no other NormalStage that can reach this.
        KAS_ASSERT(getParents().size() == 1);
    }
    AbstractStage *parent = Base::arbitraryParentImpl();
    NormalStage *normalParent = dynamic_cast<NormalStage *>(parent);
    if (!normalParent) {
        KAS_ASSERT(origin == NodeType::Reducing);
        // ReductionStage.
        KAS_ASSERT(dynamic_cast<ReductionStage *>(parent));
        return parent;
    }
    if (normalParent->origin == NodeType::Reducing) {
        // Note that this is embedded! We had better jump this.
        KAS_ASSERT(normalParent->getParents().size() == 1);
        // But we must not jump directly to ReductionStage.
        // Because to do that, we must call arbitraryParent() on that, which acquires the lock.
        // Remember that we must not acquire ancestors' locks.
        // So the dirty work is left for Node to do.
    }
    return normalParent;
}

Finalizability NormalStage::experimentFinalizability(Lock& lock) {
    if (!possibleToFinalizeByExperimenting()) {
        // If proved to be not finalizable, no need to generate children.
        childrenGenerated = true;
        // We cannot propagte, because the parent is now building children.
        // Luckily our parent will first check the finalizability of this.
        determineFinalizability(Finalizability::No, false);
    } else {
        // Because we have added this in ReductionStage.
        getStats().removeEmbededRedundancy();
    }
    return getFinalizability(lock);
}

ShapeDistance NormalStage::getShapeDistanceImpl() const {
    return shapeDistance;
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
        } else if (next.type == Next::Type::Contraction) {
            return nextContractions.findTransform<Arc>(next, [this](const NextContractionSlot& slot) -> Arc {
                return { &sampler, slot.op };
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
        } else if (next.type == Next::Type::Contraction) {
            auto slot = nextContractions.getSlot(next);
            if (!slot) return std::optional<Node>();
            return std::optional<Node>(std::in_place, &sampler, slot->nextStage);
        }
        return Base::getChildImpl(next);
    });
}

bool NormalStage::canAcceptArcImpl(Arc arc) {
    if (auto ptr = arc.tryAs<ReduceOp>(); ptr) {
        // We have left ReductionStage.
        return false;
    }
    return Base::canAcceptArcImpl(arc);
}

} // namespace kas
