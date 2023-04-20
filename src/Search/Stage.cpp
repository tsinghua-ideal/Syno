#include "KAS/Search/Stage.hpp"
#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Common.hpp"
#include <fmt/core.h>


namespace kas {

std::size_t NextBound::size() const {
    return finalizeCount + repeatLikeCount + splitLikeCount + mergeLikeCount;
}

Next NextBound::get(std::size_t index) const {
    if (index < finalizeCount) {
        return Next { Next::Type::Finalize, index };
    }
    index -= finalizeCount;
    if (index < repeatLikeCount) {
        return Next { Next::Type::RepeatLike, index };
    }
    index -= repeatLikeCount;
    if (index < splitLikeCount) {
        return Next { Next::Type::SplitLike, index };
    }
    index -= splitLikeCount;
    if (index < mergeLikeCount) {
        return Next { Next::Type::MergeLike, index };
    }
    throw std::runtime_error("Invalid index for Next.");
}

Stage *StageStore::Convert(ColoredInterface *from) {
    return reinterpret_cast<Stage *>(reinterpret_cast<std::size_t>(from) - offsetof(Stage, interface));
}

Stage *StageStore::find(ColoredInterface *interface) const {
    KAS_ASSERT(std::ranges::is_sorted(interface->items, Dimension::HashLessThan{}, ColoredDimension::Projection{}), "Interface is not sorted.");
    if (auto it = interfaces.find(interface); it != interfaces.end()) {
        return Convert(*it);
    } else {
        return nullptr;
    }
}

bool StageStore::insert(Stage *stage) {
    return interfaces.insert(&stage->interface).second;
}

StageStore::~StageStore() {
    for (auto interface: interfaces) {
        delete Convert(interface);
    }
}

StageStore& Stage::getStageStore() {
    return sampler.getStageStore();
}

void Stage::guard() {
    if (nexts.has_value()) {
        return;
    }
    const BindingContext& ctx = sampler.getBindingContext();
    const SampleOptions& options = sampler.getOptions();
    DimensionStore& store = sampler.getDimStore();

    for (auto g = FinalizeOp::Generate(interface, colors, { .ctx = ctx, .desired = sampler.getInputShape(), .maximumTensors = options.maximumTensors }); auto& f: g)
        finalizes.emplace_back(std::move(f), nullptr);

    std::vector<const RepeatLikeOp *> nextRepeatLikes;
    std::vector<const SplitLikeOp *> nextSplitLikes;
    std::vector<const MergeLikeOp *> nextMergeLikes;

    if (depth < options.depth) {
        // Keep dimensionality, by applying `RepeatLikeOp`^{-1}s.
        // Shift^{-1}, TODO
        // Stride^{-1}
        std::ranges::move(StrideOp::Generate(store, interface, colors), std::back_inserter(nextRepeatLikes));

        // Try decreasing dimensionality, by applying `SplitLikeOp`^{-1}s.
        // Split^{-1}
        std::ranges::move(SplitOp::Generate(store, interface, colors, { .dimLowerBound = options.dimLowerBound }), std::back_inserter(nextSplitLikes));
        // Unfold^{-1}
        std::ranges::move(UnfoldOp::Generate(store, interface, colors, { .ctx = ctx, .dimLowerBound = options.dimLowerBound }), std::back_inserter(nextSplitLikes));

        // Try increasing dimensionality, by applying `MergeLikeOp`^{-1}s.
        // Merge^{-1}
        std::ranges::move(MergeOp::Generate(store, interface, colors, { .ctx = ctx, .dimUpperBound = options.dimUpperBound }), std::back_inserter(nextMergeLikes));
        // Share^{-1}
        std::ranges::move(ShareOp::Generate(store, interface, colors, { .ctx = ctx, .dimUpperBound = options.dimUpperBound }), std::back_inserter(nextMergeLikes));
    }

    for (auto& op: nextRepeatLikes) {
        auto stage = getNext<RepeatLikeOp>(op);
        if (stage) this->nextRepeatLikes.emplace_back(op, stage);
        else removeOp<RepeatLikeOp>(op);
    }
    for (auto& op: nextSplitLikes) {
        auto stage = getNext<SplitLikeOp>(op);
        if (stage) this->nextSplitLikes.emplace_back(op, stage);
        else removeOp<SplitLikeOp>(op);
    }
    for (auto& op: nextMergeLikes) {
        auto stage = getNext<MergeLikeOp>(op);
        if (stage) this->nextMergeLikes.emplace_back(op, stage);
        else removeOp<MergeLikeOp>(op);
    }

    nexts = NextBound {
        .finalizeCount = finalizes.size(),
        .repeatLikeCount = this->nextRepeatLikes.size(),
        .splitLikeCount = this->nextSplitLikes.size(),
        .mergeLikeCount = this->nextMergeLikes.size(),
    };
}

TensorView *Stage::getFinalize(std::size_t index) {
    auto& [op, tensorView] = finalizes.at(index);
    if (!tensorView) {
        KAS_DEBUG("Building TensorView from Finalization. Iterator graph:\n{}", GraphvizGen(op.tensors, sampler.getBindingContext()).print("kernel"));
        tensorView = op.buildTensorView();
    }
    return tensorView.get();
}

std::size_t Stage::countChildren() {
    guard();
    return nexts->size();
}

bool Stage::isFinal(std::size_t index) {
    guard();
    return nexts->get(index).type == Next::Type::Finalize;
}

std::variant<Stage *, TensorView *> Stage::next(std::size_t index) {
    guard();
    Next next = nexts->get(index);
    switch (next.type) {
        case Next::Type::Finalize:
            return getFinalize(next.index);
        case Next::Type::RepeatLike:
            return nextRepeatLikes.at(next.index).second;
        case Next::Type::SplitLike:
            return nextSplitLikes.at(next.index).second;
        case Next::Type::MergeLike:
            return nextMergeLikes.at(next.index).second;
    }
    KAS_UNREACHABLE();
}

std::string Stage::opType(std::size_t index) {
    guard();
    Next next = nexts->get(index);
    switch (next.type) {
    case Next::Type::Finalize:
        return "Finalize";
    case Next::Type::RepeatLike:
        return fmt::format("{}", nextRepeatLikes[next.index].first->getType());
    case Next::Type::SplitLike:
        return fmt::format("{}", nextSplitLikes[next.index].first->getType());
    case Next::Type::MergeLike:
        return fmt::format("{}", nextMergeLikes[next.index].first->getType());
    }
    KAS_UNREACHABLE();
}

std::string Stage::opDescription(std::size_t index) {
    guard();
    const BindingContext& ctx = sampler.getBindingContext();
    Next next = nexts->get(index);
    switch (next.type) {
        case Next::Type::Finalize:
            return "Finalize";
        case Next::Type::RepeatLike: {
            const auto& n = nextRepeatLikes[next.index].first;
            return fmt::format(
                "{} {} -> {}",
                n->getType(),
                n->getInput().description(ctx),
                n->output.description(ctx)
            );
        }
        case Next::Type::SplitLike: {
            const auto& n = nextSplitLikes[next.index].first;
            return fmt::format(
                "{} {} -> {}, {}",
                n->getType(),
                n->getInput().description(ctx),
                n->outputLhs.description(ctx),
                n->outputRhs.description(ctx)
            );
        }
        case Next::Type::MergeLike: {
            const auto& n = nextMergeLikes[next.index].first;
            auto [inputLhs, inputRhs] = n->getInputs();
            return fmt::format(
                "{} {}, {} -> {}",
                n->getType(),
                inputLhs.description(ctx),
                inputRhs.description(ctx),
                n->output.description(ctx)
            );
        }
    }
    KAS_UNREACHABLE();
}

} // namespace kas
