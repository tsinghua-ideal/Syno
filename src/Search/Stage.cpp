#include "KAS/Search/Stage.hpp"
#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Common.hpp"
#include <fmt/core.h>


namespace kas {

std::size_t StageStore::Hash::operator()(const ColoredInterface& interface) const noexcept {
    std::size_t h = interface.items.size();
    for (const auto& dim: interface.items) {
        HashCombine(h, dim.dimension.hash());
    }
    return h;
}

std::size_t StageStore::Hash::operator()(const Stage * stage) const noexcept {
    return (*this)(stage->getInterface());
}

bool StageStore::Equal::operator()(const ColoredInterface& lhs, const ColoredInterface& rhs) const noexcept {
    return std::ranges::equal(lhs.items, rhs.items, std::equal_to<Dimension>{}, ColoredDimension::Projection{}, ColoredDimension::Projection{});
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
    KAS_ASSERT(std::ranges::is_sorted(interface.items, Dimension::HashLessThan{}, ColoredDimension::Projection{}), "Interface is not sorted.");
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

void Stage::guard() {
    if (childrenGenerated) {
        return;
    }
    const BindingContext& ctx = sampler.getBindingContext();
    const SampleOptions& options = sampler.getOptions();
    DimensionStore& store = sampler.getDimStore();

    auto accumulate = [](const auto& slot) {
        return slot.toNext();
    };

    // First add finalizations.
    for (auto g = FinalizeOp::Generate(interface, colors, { .ctx = ctx, .desired = sampler.getInputShape(), .maximumTensors = options.maximumTensors }); auto& f: g)
        nextFinalizations.emplace_back(f.getHash(), std::move(f), nullptr);
    std::ranges::sort(nextFinalizations, std::less{}, &NextFinalizeSlot::key);
    std::ranges::move(nextFinalizations | std::views::transform(accumulate), std::back_inserter(nexts));

    // Wrap the generated Op in a NextOpSlot.
    auto nextOpProcessor = [&]<typename Op>(const Op *op) -> NextOpSlot<Op> {
        return { op->opHash(), op, getNextOp<Op>(op) };
    };
    // Filter out illegal transforms.
    auto nextOpFilter = []<typename Op>(const NextOpSlot<Op>& opSlot) -> bool {
        return opSlot.nextStage != nullptr;
    };
    auto add = [&]<typename Op>(const std::vector<const Op *>& newOps) {
        std::ranges::move(
            newOps
            | std::views::transform(nextOpProcessor) | std::views::filter(nextOpFilter),
            std::back_inserter(nextOpStores.get<Op>())
        );
        // Sort according to keys for binary search.
        std::ranges::sort(nextOpStores.get<Op>(), std::less{}, &NextOpSlot<Op>::key);
        // Add to all nexts.
        std::ranges::move(nextOpStores.get<Op>() | std::views::transform(accumulate), std::back_inserter(nexts));
    };

    if (depth < options.depth) {
        // Keep dimensionality, by applying `RepeatLikeOp`^{-1}s.
        // Shift^{-1}, TODO
        // Stride^{-1}
        add(StrideOp::Generate(store, interface, colors, { .ctx = ctx }));

        // Try decreasing dimensionality, by applying `SplitLikeOp`^{-1}s.
        // Split^{-1}
        add(SplitOp::Generate(store, interface, colors, { .dimLowerBound = options.dimLowerBound }));
        // Unfold^{-1}
        add(UnfoldOp::Generate(store, interface, colors, { .ctx = ctx, .dimLowerBound = options.dimLowerBound }));

        // Try increasing dimensionality, by applying `MergeLikeOp`^{-1}s.
        // Merge^{-1}
        add(MergeOp::Generate(store, interface, colors, { .ctx = ctx, .dimUpperBound = options.dimUpperBound }));
        // Share^{-1}
        add(ShareOp::Generate(store, interface, colors, { .ctx = ctx, .dimUpperBound = options.dimUpperBound }));
    }

    childrenGenerated = true;
}

TensorView *Stage::getFinalize(std::size_t key) {
    auto& [_, op, tensorView] = getChildFinalizeSlot(key);
    if (!tensorView) {
        KAS_DEBUG("Building TensorView from Finalization. Iterator graph:\n{}", GraphvizGen(op.tensors, sampler.getBindingContext()).print("kernel"));
        tensorView = op.buildTensorView();
    }
    return tensorView.get();
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
