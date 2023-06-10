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
    std::size_t h = interface.size();
    for (const auto& dim: interface.toDimensions()) {
        HashCombine(h, dim.hash());
    }
    return h;
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
        if (!options.disableStride) {
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
            if (!options.disableSplit) {
                add(SplitOp::Generate(store, interface, {
                    .disallowDiscontinuousView = options.disallowDiscontinuousView,
                    .disallowSplitRAboveUnfold = options.disallowSplitRAboveUnfold,
                    .disallowSplitRAboveStride = options.disallowSplitRAboveStride,
                }));
            }
            // Unfold^{-1}
            if (!options.disableUnfold) {
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
            if (!options.disableMerge) {
                add(MergeOp::Generate(store, interface, {
                    .ctx = ctx,
                    .minimumRatio = options.minimumMergeRatio,
                    .disallowMergeWithLargeBlockAboveStride = options.disallowMergeWithLargeBlockAboveStride,
                    .disallowMergeWithLargeBlockAboveUnfold = options.disallowMergeWithLargeBlockAboveUnfold,
                }));
            }
            // Share^{-1}
            if (!options.disableShare) {
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

bool Stage::possibleToFinalize() const {
    ++CountFinalizabilityCheckInvocations;

    const auto& ctx = sampler.getBindingContext();
    const auto& options = sampler.getOptions();
    const std::size_t remainingSteps = remainingDepth();

    // First, check for strided dimensions. They must be absorbed by UnfoldOp before Finalization.
    std::size_t stridedDims = std::ranges::count_if(interface, [](const ColoredDimension& cdim) { return cdim.color.isDataDiscarding(); });
    if (stridedDims > remainingSteps) {
        // We can remove 1 strided dimension per step.
        ++CountTooManyStridedDims;
        return false;
    }

    // Next, check that there exists a coloring for weight tensors. Note that this check is conservative.
    auto weightTensorDims =
        interface
        | std::views::filter([](const ColoredDimension& cDim) {
            return cDim.deduceOrigin() == ColoredDimension::Origin::Weight;
        });
    switch (options.maximumTensors) {
    case 1: {
        // There must be exactly 1 weight tensor.
        if (std::ranges::distance(weightTensorDims) > 0) {
            ++CountTooManyWeights;
            return false;
        }
        break;
    }
    case 2: {
        // We have to check that the colors won't conflict.
        Color color;
        for (const auto& cDim: weightTensorDims) {
            if (!color.disjoint(cDim.color)) {
                ++CountTooManyWeights;
                return false;
            }
            color.merge(cDim.color);
        }
        break;
    }
    default:
        KAS_CRITICAL("Unsupported maximumTensors: {}", options.maximumTensors);
    }

    // Then, check whether there are enough elements in the input tensor.
    auto inputTensorCandidates =
        interface
        | std::views::filter([](const ColoredDimension& cDim) { return cDim.deduceOrigin() == ColoredDimension::Origin::Input || cDim.deduceOrigin() == ColoredDimension::Origin::BothPossible; })
        | std::views::transform(&ColoredDimension::dimension);
    std::unordered_map<Size, int> counts;
    auto elements = Size::Identity(ctx);
    for (const auto& dim: inputTensorCandidates) {
        // Take the product to find the total elements.
        elements = elements * dim.size();
        counts[dim.size()] += 1;
    }
    // Check whether there are enough elements.
    auto quotient = elements.testDividedBy(sampler.getInputShape().totalSize());
    if (!quotient || *quotient == Size::Trait::IllegalCoefficient) {
        ++CountTooFewElementsInInputTensor;
        return false;
    }

    // At last, try matching the dimensions, which is just experimental Finalization.
    std::vector<const Size *> unfulfilled;
    for (const auto& required: sampler.getInputShape()) {
        auto it = counts.find(required);
        if (it != counts.end()) {
            if (it->second == 1) {
                counts.erase(it);
            } else {
                it->second -= 1;
            }
        } else {
            unfulfilled.push_back(&required);
        }
    }
    if ((unfulfilled.size() + 1) / 2 + stridedDims > remainingSteps) {
        // Merge can possibly fulfill 2 dimensions per step. So this is most conservative.
        ++CountShapeDeviatesTooMuch;
        return false;
    } else if (unfulfilled.size() + stridedDims > remainingSteps) {
        // We need at least 1 Merge tree to make things work. Enumerate.
        auto enumerateSizes = [&](const auto& self, const Size& previousSize, std::size_t nextIndex) -> bool {
            if (nextIndex == unfulfilled.size()) {
                return counts.contains(previousSize);
            } else {
                return self(self, previousSize, nextIndex + 1)
                    || self(self, previousSize * *unfulfilled[nextIndex], nextIndex + 1);
            }
        };
        if (enumerateSizes(enumerateSizes, Size::Identity(ctx), 0)) {
            return true;
        }
        ++CountShapeDeviatesTooMuch;
        return false;
    } else {
        // We still have enough steps.
        return true;
    }
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
