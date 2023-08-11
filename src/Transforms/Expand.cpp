#include "KAS/Transforms/Expand.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

GraphHandle ExpandOp::applyToInterface(const GraphHandle& interface) const {
    return interface.moveToExpansions(this);
}

std::vector<const ExpandOp *> ExpandOp::Generate(PrimitiveOpStore& store, const GraphHandle& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    // First, check if there are too many Expand's.
    const auto currentExpansion = std::transform_reduce(
        interface.getExpansions().begin(),
        interface.getExpansions().end(),
        Size::Identity(options.ctx),
        std::multiplies<>{},
        [](const Expand *expansion) { return expansion->output.size(); }
    );

    // Here we only allow repeat semantics.
    using T = DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> allows { T::MergeR };
    if (!options.disallowTile) {
        allows.emplace_back(T::MergeL);
    }
    if (!options.disallowMergeInputAndWeight) {
        allows.emplace_back(T::ShareL); // We later look for the Merge pattern.
    }
    auto plausible = interface.filterIn(allows);

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
            // Yes, it is.
        }
        if (
            options.maxExpansionMultiplier &&
            // Do not make expansions too large.
            (currentExpansion * dim.size()).upperBoundEst(options.ctx) > options.maxExpansionMultiplier
        ) {
            continue;
        }
        ++CountSuccessfulGenerations;
        res.emplace_back(store.get<ExpandOp>(dim));
    }
    CountDisallowedAttempts += interface.getDimensions().size() - countPlausible;
    return res;
}

} // namespace kas
