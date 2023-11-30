#include "KAS/Core/Graph.hpp"
#include "KAS/Transforms/Expand.hpp"
#include "KAS/Transforms/OperationStore.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

bool ExpandOp::isEqual(const Operation& other) const {
    return output == static_cast<const ExpandOp&>(other).output;
}

std::size_t ExpandOp::initialHash() const noexcept {
    return DimensionTypeHash(Type);
}

std::size_t ExpandOp::opHash() const noexcept {
    std::size_t h = initialHash();
    HashCombineRaw(h, output.hash());
    return h;
}

bool ExpandOp::canApplyToInterface(const GraphHandle& interface) const {
    return interface.contains(output);
}

void ExpandOp::applyToInterface(GraphHandle& interface) const {
    interface.moveToExpansions(this);
}

std::string ExpandOp::description(const BindingContext& ctx) const {
    return fmt::format("-> {}", output.description(ctx));
}
std::string ExpandOp::descendantsDescription(const BindingContext& ctx) const {
    return fmt::format("-> {}", output.descendantsDescription(ctx));
}

Reshape::BlockSet ExpandOp::GetReshapeBlocks(ReshapeCanonicalizer& canonicalizer, const Graph& graph) {
    graph.accept(canonicalizer);
    return Reshape::BlockSet::From(
        graph.getTopmost().getExpansions()
        | std::views::transform([&](const Expand *expand) -> decltype(auto) {
            return canonicalizer.at(expand->output);
        })
    );
}

std::vector<const ExpandOp *> ExpandOp::Generate(OperationStore& store, const Topmost& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    // Fast path.
    if (options.maxExpansionRepeatMultiplier == 1) {
        return {};
    }

    const BindingContext& ctx = options.ctx;
    const Graph& graph = options.graph;

    ReshapeCanonicalizer canonicalizer;
    Reshape::BlockSet combined;

    // We need to check if there are too many Expand's.
    auto currentExpansionRepeat = Size::Identity(ctx);
    for (auto expansion: interface.getExpansions()) {
        if (expansion->output.is(DimensionType::Merge)) {
            // Expand + Merge == Repeat.
            currentExpansionRepeat = currentExpansionRepeat * expansion->output.size();
        }
    }

    // Here we only allow repeat semantics.
    using T = DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> allows { T::MergeR };
    if (!options.disallowTile) {
        allows.emplace_back(T::MergeL);
    }
    auto plausible = interface.filterIn(std::move(allows));

    std::vector<const ExpandOp *> res;
    CountGenerateAttempts += interface.getDimensions().size();
    std::size_t countPlausible = 0;
    for (auto&& dim: plausible) {
        ++countPlausible;
        // Check if the other is undesired.
        // if (canonicalizer.at(dim).isAdjacentTo(combined)) {
        //     continue;
        // }

        // Certain cases are redundant.
        const auto& dimMerge = dim.as<MergeOp::Input>();
        Dimension outputDim = dimMerge.getOp()->output;
        if (outputDim.is(DimensionType::Merge)) {
            // This must be a tile.
            if (options.disallowTile) {
                continue;
            }
        }

        if (
            options.maxExpansionRepeatMultiplier &&
            // Do not make expansions too large.
            (currentExpansionRepeat * dim.size()).upperBoundEst(ctx) > options.maxExpansionRepeatMultiplier
        ) {
            continue;
        }
        ++CountSuccessfulGenerations;
        res.emplace_back(store.get<ExpandOp>(dim));
    }
    CountDisallowedAttempts += interface.getDimensions().size() - countPlausible;
    return res;
}

RepeatOp::RepeatOp(const ExpandOp& expandOp, const MergeOp& mergeOp):
    expandOp { expandOp },
    mergeOp { mergeOp },
    input { expandOp.output.as<MergeOp::Input>().getOther() },
    kind { expandOp.output.as<MergeOp::Input>().getOrder() == Order::Right ? Repeat : Tile },
    output { mergeOp.output }
{
    KAS_ASSERT(expandOp.output.as<MergeOp::Input>().getOp() == &mergeOp);
}

std::string RepeatOp::description(const BindingContext& ctx) const {
    return fmt::format("{} -> {}", input.description(ctx), output.description(ctx));
}

} // namespace kas
