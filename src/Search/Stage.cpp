#include <fmt/core.h>
#include <unordered_map>

#include "KAS/Search/Stage.hpp"
#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::size_t StageStore::Hash::operator()(const ColoredInterface& interface) const noexcept {
    return std::hash<Interface>{}(interface.toDimensions());
}

std::size_t StageStore::Hash::operator()(const Stage * stage) const noexcept {
    return (*this)(stage->getInterface());
}

bool StageStore::Equal::operator()(const ColoredInterface& lhs, const ColoredInterface& rhs) const noexcept {
    return std::ranges::equal(lhs.toDimensions(), rhs.toDimensions());
}
bool StageStore::Equal::operator()(const ColoredInterface& lhs, const Stage *rhs) const noexcept {
    return (*this)(lhs, rhs->getInterface());
}
bool StageStore::Equal::operator()(const Stage *lhs, const ColoredInterface& rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs);
}
bool StageStore::Equal::operator()(const Stage *lhs, const Stage *rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs->getInterface());
}

Stage *StageStore::find(const ColoredInterface& interface) const {
    KAS_ASSERT(std::ranges::is_sorted(interface.toDimensions(), Dimension::HashLessThan{}), "Interface is not sorted.");
    if (auto it = interfaces.find(interface); it != interfaces.end()) {
        return *it;
    } else {
        return nullptr;
    }
}

bool StageStore::insert(Stage *stage) {
    return interfaces.insert(stage).second;
}

StageStore::~StageStore() {
    for (auto interface: interfaces) {
        delete interface;
    }
}

StageStore& Stage::getStageStore() {
    return sampler.getStageStore();
}

std::size_t Stage::remainingDepth() const {
    return sampler.getOptions().depth - depth;
}

void Stage::guard() {
    if (childrenGenerated) {
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
    nexts = nextFinalizations.toNexts();

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
                /* We need to call removeOp for deleted Op's and removeStage for deleted Stage's. TODO */
                return !slot.nextStage->possibleToFinalize();
            });
        std::ranges::move(nextOpStores.get<Op>().toNexts(), std::back_inserter(nexts));
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
}

std::shared_ptr<TensorView> Stage::getFinalize(std::size_t key) {
    auto& slot = getChildFinalizeSlot(key);
    KAS_DEBUG("Building TensorView from Finalization.");
    return slot.finalization.buildTensorView(sampler.getFixedDimensions());
}

Stage::Stage(Sampler& sampler, const std::vector<const MapReduceOp *>& reductions):
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
}

Stage::Stage(Sampler& sampler, ColoredInterface&& interface, const Stage& old, Next::Type delta):
    interface { std::move(interface) },
    sampler { sampler },
    depth { old.depth + 1 },
    existingOps { old.existingOps }
{
    existingOps[static_cast<std::size_t>(delta)] += 1;
}

bool Stage::possibleToFinalize() const {
    if (possibleToFinalizeCache) {
        return *possibleToFinalizeCache;
    }

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
        possibleToFinalizeCache = false;
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
        possibleToFinalizeCache = false;
        return false;
    }

    possibleToFinalizeCache = true;
    return true;
}

std::size_t Stage::countChildren() {
    guard();
    return nexts.size();
}

NextFinalizeSlot& Stage::getChildFinalizeSlot(std::size_t key) {
    guard();
    return nextFinalizations.getSlot(key);
}

Node Stage::getChild(Next next) {
    guard();
    switch (next.type) {
    case Next::Type::Shift: return { &sampler, getChildSlot<ShiftOp>(next.key).nextStage };
    case Next::Type::Stride: return { &sampler, getChildSlot<StrideOp>(next.key).nextStage };
    case Next::Type::Split: return { &sampler, getChildSlot<SplitOp>(next.key).nextStage };
    case Next::Type::Unfold: return { &sampler, getChildSlot<UnfoldOp>(next.key).nextStage };
    case Next::Type::Merge: return { &sampler, getChildSlot<MergeOp>(next.key).nextStage };
    case Next::Type::Share: return { &sampler, getChildSlot<ShareOp>(next.key).nextStage };
    case Next::Type::Finalize: return { &sampler, getFinalize(next.key) };
    default: KAS_UNREACHABLE("Invalid Next {}.", next.toString());
    }
}

std::string Stage::description() const {
    return DimensionArrayToString(interface.toDimensions(), sampler.getBindingContext());
}

} // namespace kas
