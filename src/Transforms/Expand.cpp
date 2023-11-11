#include "KAS/Core/Graph.hpp"
#include "KAS/Transforms/Expand.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

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

bool ExpandOp::operator==(const ExpandOp& other) const noexcept {
    return output == other.output;
}

std::string ExpandOp::description(const BindingContext& ctx) const {
    return fmt::format("-> {}", output.description(ctx));
}
std::string ExpandOp::descendantsDescription(const BindingContext& ctx) const {
    return fmt::format("-> {}", output.descendantsDescription(ctx));
}

std::vector<const ExpandOp *> ExpandOp::Generate(PrimitiveOpStore& store, const Topmost& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    const BindingContext& ctx = options.ctx;

    // We need to check if there are too many Expand's.
    auto currentExpansionRepeat = Size::Identity(ctx);
    // Also record all the weight dims brought by Expand. Weight dims cannot be Merge'd with weight dims.
    // Moreover, I cannot see what is useful for Merging expanded dims with expanded dims or weight dims.
    // I think expanded dims are only useful when they merge with data dims. TODO: think over this.
    Graph::DimensionSet expandedDims;
    for (auto expansion: interface.getExpansions()) {
        if (expansion->output.is(DimensionType::Merge)) {
            // Expand + Merge == Repeat.
            currentExpansionRepeat = currentExpansionRepeat * expansion->output.size();
            expandedDims.emplace(expansion->output);
        } else {
            // Merge input and weight, or else.
            Dimension weightDim = expansion->output;
            KAS_ASSERT(weightDim.is(DimensionTypeWithOrder::ShareL));
            // Note that the weight dim can be shared by multiple weights.
            // So we have to find the bottommost Share.
            while (weightDim.is(DimensionTypeWithOrder::ShareL)) {
                weightDim = weightDim.as<ShareOp::Input>().getOp()->output;
            }
            auto bottommostType = weightDim.type();
            if (bottommostType == DimensionType::Merge) {
                expandedDims.emplace(weightDim);
            } else {
                // Multiple weights contribute to a single dim.
                // Currently we only allow this for Iterator and Reduce.
                KAS_ASSERT(bottommostType == DimensionType::Iterator || bottommostType == DimensionType::Reduce);
            }
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
        const auto& dimMerge = dim.as<MergeOp::Input>();
        auto otherInputOfMerge = dimMerge.getOther();
        // Check if the other is undesired.
        if (expandedDims.contains(otherInputOfMerge)) {
            continue;
        }

        // Certain cases are redundant.
        Dimension outputDim = dimMerge.getOp()->output;
        if (outputDim.is(DimensionType::Merge)) {
            // This must be a tile.
            if (options.disallowTile) {
                continue;
            }
            // Get the bottommost output dim.
            while (outputDim.is(DimensionType::Merge)) {
                outputDim = outputDim.as<MergeOp::Input>().getOp()->output;
            }
        }
        if (outputDim.type() == DimensionType::Iterator) {
            // Data replication.
            continue;
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
