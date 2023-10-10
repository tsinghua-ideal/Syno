#include "KAS/Transforms/Expand.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

GraphHandle ExpandOp::applyToInterface(const GraphHandle& interface) const {
    return interface.moveToExpansions(this);
}

std::vector<const ExpandOp *> ExpandOp::Generate(PrimitiveOpStore& store, const GraphHandle& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    // We need to check if there are too many Expand's.
    auto currentExpansionRepeat = Size::Identity(options.ctx);
    auto currentExpansionMerge = Size::Identity(options.ctx);
    for (auto expansion: interface.getExpansions()) {
        if (expansion->output.is(DimensionType::Merge)) {
            // Expand + Merge == Repeat.
            currentExpansionRepeat = currentExpansionRepeat * expansion->output.size();
        } else {
            // Merge input and weight.
            KAS_ASSERT(expansion->output.is(DimensionType::Share));
            currentExpansionMerge = currentExpansionMerge * expansion->output.size();
        }
    }

    // Here we only allow repeat semantics.
    using T = DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> allows { T::MergeR };
    if (!options.disallowTile) {
        allows.emplace_back(T::MergeL);
    }
    if (!options.disallowMergeInputAndWeight) {
        allows.emplace_back(T::ShareL); // We later look for the Merge pattern.
    }
    auto plausible = interface.filterIn(std::move(allows));

    std::vector<const ExpandOp *> res;
    CountGenerateAttempts += interface.getDimensions().size();
    std::size_t countPlausible = 0;
    for (auto&& dim: plausible) {
        ++countPlausible;
        // If this is not a simple repeat,
        if (auto share = dim.tryAs<ShareOp::Input>(); share) {
            // this may be a share across tensors.
            if (share->getOp()->output.type() != DimensionType::Merge) {
                continue;
            }
            // Yes, it is. But check if this the other input of Merge is also an Expand.
            auto otherInputOfMerge = share->getOp()->output.as<MergeOp::Input>().getOther();
            if (std::ranges::any_of(interface.getExpansions(), [&](const Expand *expand) {
                return expand->output == otherInputOfMerge;
            })) {
                // This is ridiculous: we are getting the entire component from weights!
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
        } else {
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

} // namespace kas
