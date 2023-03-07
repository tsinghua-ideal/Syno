#include "KAS/Search/Stage.hpp"
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

Stage *StageStore::Convert(Interface *from) {
    return reinterpret_cast<Stage *>(reinterpret_cast<std::size_t>(from) - offsetof(Stage, interface));
}

Stage *StageStore::find(Interface *interface) const {
    KAS_ASSERT(std::ranges::is_sorted(*interface), "Interface is not sorted.");
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

void Stage::guard() {
    if (nexts.has_value()) {
        return;
    }
    const BindingContext& ctx = sampler.getBindingContext();
    const SampleOptions& options = sampler.getOptions();
    DimensionStore& store = sampler.getDimStore();
    for (auto g = FinalizeOp::Generate(interface, { .ctx = ctx, .desired = sampler.getInputShape() }); auto& f: g)
        finalizes.emplace_back(std::move(f), nullptr);
    if (depth < options.depth) {
        // Keep dimensionality, by applying `RepeatLikeOp`^{-1}s.
        // Shift^{-1}, TODO
        // Stride^{-1}
        for (auto g = StrideOp::Generate(store, interface); auto& s: g)
            nextRepeatLikes.emplace_back(std::move(s), nullptr);

        // Try decreasing dimensionality, by applying `SplitLikeOp`^{-1}s.
        // Split^{-1}
        for (auto g = SplitOp::Generate(store, interface, { .dimLowerBound = options.dimLowerBound }); auto& s: g)
            nextSplitLikes.emplace_back(std::move(s), nullptr);
        // Unfold^{-1}
        for (auto g = UnfoldOp::Generate(store, interface, { .ctx = ctx, .dimLowerBound = options.dimLowerBound }); auto& u: g)
            nextSplitLikes.emplace_back(std::move(u), nullptr);

        // Try increasing dimensionality, by applying `MergeLikeOp`^{-1}s.
        // Merge^{-1}
        for (auto g = MergeOp::Generate(store, interface, { .ctx = ctx, .dimUpperBound = options.dimUpperBound }); auto& m: g)
            nextMergeLikes.emplace_back(std::move(m), nullptr);
        // Share^{-1}
        for (auto g = ShareOp::Generate(store, interface, { .ctx = ctx, .dimUpperBound = options.dimUpperBound }); auto& s: g)
            nextMergeLikes.emplace_back(std::move(s), nullptr);
    }
    nexts = NextBound {
        .finalizeCount = finalizes.size(),
        .repeatLikeCount = nextRepeatLikes.size(),
        .splitLikeCount = nextSplitLikes.size(),
        .mergeLikeCount = nextMergeLikes.size(),
    };
}

TensorView *Stage::getFinalize(std::size_t index) {
    auto& [op, stage] = finalizes[index];
    if (!stage) {
        stage = op.buildTensorView();
    }
    return stage.get();
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
            return getNext<RepeatLikeOp>(sampler.getStageStore(), next.index);
        case Next::Type::SplitLike:
            return getNext<SplitLikeOp>(sampler.getStageStore(), next.index);
        case Next::Type::MergeLike:
            return getNext<MergeLikeOp>(sampler.getStageStore(), next.index);
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
        return DimensionTypeDescription(nextRepeatLikes[next.index].first->getType());
    case Next::Type::SplitLike:
        return DimensionTypeDescription(nextSplitLikes[next.index].first->getType());
    case Next::Type::MergeLike:
        return DimensionTypeDescription(nextMergeLikes[next.index].first->getType());
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
                DimensionTypeDescription(n->getType()),
                n->getInput().description(ctx),
                n->output.description(ctx)
            );
        }
        case Next::Type::SplitLike: {
            const auto& n = nextSplitLikes[next.index].first;
            return fmt::format(
                "{} {} -> {}, {}",
                DimensionTypeDescription(n->getType()),
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
                DimensionTypeDescription(n->getType()),
                inputLhs.description(ctx),
                inputRhs.description(ctx),
                n->output.description(ctx)
            );
        }
    }
    KAS_UNREACHABLE();
}

} // namespace kas
