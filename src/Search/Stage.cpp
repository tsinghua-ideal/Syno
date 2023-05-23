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

    auto accumulate = [](const auto& slot) {
        return slot.toNext();
    };

    // First add finalizations.
    for (auto g = FinalizeOp::Generate(interface, graph, {
        .ctx = ctx,
        .desired = sampler.getInputShape(),
        .maximumTensors = options.maximumTensors,
    }); auto& f: g)
        nextFinalizations.emplace_back(f.getHash(), std::move(f), nullptr);
    std::ranges::sort(nextFinalizations, std::less{}, &NextFinalizeSlot::key);
    std::ranges::move(nextFinalizations | std::views::transform(accumulate), std::back_inserter(nexts));

    // Wrap the generated Op in a NextOpSlot.
    auto nextOpProcessor = [&]<typename Op>(const Op *op) -> NextOpSlot<Op> {
        return { op->opHash(), op, getNextOp<Op>(op) };
    };
    auto add = [&]<typename Op>(const std::vector<const Op *>& newOps) {
        std::ranges::move(
            newOps
            | std::views::filter([&](const Op *op) { return ShareOp::IsSharedDimensionCanonical(op, graph); }) /* We need to call removeOp for deleted Op's. TODO */
            | std::views::transform(nextOpProcessor)
            | std::views::filter([&](const NextOpSlot<Op>& slot) { return slot.nextStage->possibleToFinalize(); }) /* We need to call removeOp for deleted Op's and removeStage for deleted Stage's. TODO */,
            std::back_inserter(nextOpStores.get<Op>())
        );
        // Sort according to keys for binary search.
        std::ranges::sort(nextOpStores.get<Op>(), std::less{}, &NextOpSlot<Op>::key);
        // Add to all nexts.
        std::ranges::move(nextOpStores.get<Op>() | std::views::transform(accumulate), std::back_inserter(nexts));
    };

    if (remainingDepth() > 0) {
        // Keep dimensionality, by applying `RepeatLikeOp`^{-1}s.
        // Shift^{-1}, TODO
        // Stride^{-1}
        add(StrideOp::Generate(store, interface, {
            .ctx = ctx,
            .maxStridedDimSize = options.maxStridedDimSize,
            .disallowStrideAboveSplit = options.disallowStrideAboveSplit,
            .disallowStrideAboveMergeR = options.disallowStrideAboveMergeR,
        }));

        // Try decreasing dimensionality, by applying `SplitLikeOp`^{-1}s.
        if (interface.size() > options.dimLowerBound) {
            // Split^{-1}
            add(SplitOp::Generate(store, interface, {
                .disallowDiscontinuousView = options.disallowDiscontinuousView,
                .disallowSplitRAboveUnfold = options.disallowSplitRAboveUnfold,
                .disallowSplitRAboveStride = options.disallowSplitRAboveStride,
            }));
            // Unfold^{-1}
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

        // Try increasing dimensionality, by applying `MergeLikeOp`^{-1}s.
        if (interface.size() < options.dimUpperBound) {
            // Merge^{-1}
            add(MergeOp::Generate(store, interface, {
                .ctx = ctx,
                .minimumRatio = options.minimumMergeRatio,
                .disallowMergeWithLargeBlockAboveStride = options.disallowMergeWithLargeBlockAboveStride,
                .disallowMergeWithLargeBlockAboveUnfold = options.disallowMergeWithLargeBlockAboveUnfold,
            }));
            // Share^{-1}
            add(ShareOp::Generate(store, interface, {
                .ctx = ctx,
                .maximumTensors = options.maximumTensors,
            }));
        }
    }

    childrenGenerated = true;
}

TensorView *Stage::getFinalize(std::size_t key) {
    auto& [_, op, tensorView] = getChildFinalizeSlot(key);
    if (!tensorView) {
        KAS_DEBUG("Building TensorView from Finalization.");
        tensorView = op.buildTensorView(sampler.getFixedDimensions());
    }
    return tensorView.get();
}

bool Stage::possibleToFinalize() const {
    ++CountFinalizabilityCheckInvocations;

    const auto& ctx = sampler.getBindingContext();
    const std::size_t remainingSteps = remainingDepth();

    // First, check for strided dimensions. They must be absorbed by UnfoldOp before Finalization.
    std::size_t stridedDims = std::ranges::count_if(interface, [](const ColoredDimension& cdim) { return cdim.color.isDataDiscarding(); });
    if (stridedDims > remainingSteps) {
        // We can remove 1 strided dimension per step.
        ++CountTooManyStridedDims;
        return false;
    }

    // Then, check whether there are enough elements in the input tensor.
    auto inputTensorCandidates =
        interface
        | std::views::filter([](const ColoredDimension& cdim) { return cdim.color.countRightTags() == 0 && !cdim.color.isDataDiscarding(); })
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
    if (unfulfilled.size() > remainingSteps * 2) {
        // Merge can possibly fulfill 2 dimensions per step. So this is most conservative.
        ++CountShapeDeviatesTooMuch;
        return false;
    } else if (unfulfilled.size() > remainingSteps) {
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

Stage::NextFinalizeSlot& Stage::getChildFinalizeSlot(std::size_t key) {
    guard();
    auto it = std::ranges::lower_bound(nextFinalizations, key, std::less{}, &NextFinalizeSlot::key);
    KAS_ASSERT(it != nextFinalizations.end() && it->key == key, "Specified Finalization not found.");
    return *it;
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

std::string Stage::description(const BindingContext& ctx) const {
    return DimensionArrayToString(interface.toDimensions(), ctx);
}

} // namespace kas
