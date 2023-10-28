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

GraphHandle ExpandOp::applyToInterface(const GraphHandle& interface) const {
    return interface.moveToExpansions(this);
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

Graph::DimensionSet ExpandOp::GetSharedWeightDims(const Graph& graph) {
    Graph::DimensionSet result;
    for (const Expand *expand: graph.getOpsOfType<ExpandOp>()) {
        if (auto share = expand->output.tryAs<ShareOp::Input>(); share) {
            // May be a chain of ShareOp.
            if (share->getOp()->output.is(DimensionType::Share)) {
                // OK, it is.
                while (share != nullptr) {
                    result.emplace(share->getOther());
                    share = share->getOp()->output.tryAs<ShareOp::Input>();
                }
            }
        }
    }
    return result;
}

std::vector<const ExpandOp *> ExpandOp::Generate(PrimitiveOpStore& store, const Topmost& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    // We need to check if there are too many Expand's.
    auto currentExpansionRepeat = Size::Identity(options.ctx);
    auto currentExpansionMerge = Size::Identity(options.ctx);
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
                currentExpansionMerge = currentExpansionMerge * weightDim.size();
                expandedDims.emplace(weightDim);
            } else {
                // Multiple weights contribute to a single dim.
                // Currently we only allow this for Iterator and Reduce.
                KAS_ASSERT(bottommostType == DimensionType::Iterator || bottommostType == DimensionType::Reduce);
                // And if there is only one weight, this is not allowed, since this does not need Expand.
                KAS_ASSERT(weightDim != expansion->output.as<ShareOp::Input>().getOp()->output);
            }
        }
    }

    // Here we only allow repeat semantics.
    using T = DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> allows { T::MergeR };
    if (!options.disallowTile) {
        allows.emplace_back(T::MergeL);
    }
    if (!options.disallowMergeInputAndWeight || !options.disallowShareWeights) {
        // We later look for the Merge pattern or weights-Sharing pattern.
        allows.emplace_back(T::ShareL);
    }
    auto plausible = interface.filterIn(std::move(allows));

    std::vector<const ExpandOp *> res;
    CountGenerateAttempts += interface.getDimensions().size();
    std::size_t countPlausible = 0;
    for (auto&& dim: plausible) {
        ++countPlausible;
        // If this is not a simple repeat,
        if (auto share = dim.tryAs<ShareOp::Input>(); share) {
            // this may be a merge across tensors, or weights sharing.
            auto weightDim = share->getOp()->output;
            bool isWeightsSharing = false;
            if (!options.disallowShareWeights) {
                // Get the bottommost dim, if there is weights sharing.
                while (weightDim.is(DimensionTypeWithOrder::ShareL)) {
                    weightDim = weightDim.as<ShareOp::Input>().getOp()->output;
                    isWeightsSharing = true; // Only upon chained ShareOp, this is true.
                }
            }

            auto weightDimType = weightDim.type();
            if (weightDimType == DimensionType::Merge) {
                if (options.disallowMergeInputAndWeight) {
                    continue;
                }
                // If `isWeightsSharing` is true, this is both weights-sharing and input-and-weight-merging!
                auto otherInputOfMerge = weightDim.as<MergeOp::Input>().getOther();
                // Check if the other is undesired.
                if (expandedDims.contains(otherInputOfMerge)) {
                    continue;
                }
                // Check if it exceeds the maximum.
                if (
                    options.maxExpansionMergeMultiplier &&
                    // Do not make expansions too large.
                    (currentExpansionMerge * dim.size()).upperBoundEst(options.ctx) > options.maxExpansionMergeMultiplier
                ) {
                    continue;
                }
            } else if (weightDimType == DimensionType::Iterator || weightDimType == DimensionType::Reduce) {
                // This case is only allowed in weights-sharing.
                if (!isWeightsSharing) {
                    continue;
                }
            } else {
                // Not Merge, Iterator or Reduce.
                continue;
            }
        } else {
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
                (currentExpansionRepeat * dim.size()).upperBoundEst(options.ctx) > options.maxExpansionRepeatMultiplier
            ) {
                continue;
            }
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
