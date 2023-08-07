#include "KAS/Transforms/Expand.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Share.hpp"


namespace kas {

GraphHandle ExpandOp::applyToInterface(const GraphHandle& interface) const {
    return interface.moveToExpansions(this);
}

std::vector<const ExpandOp *> ExpandOp::Generate(PrimitiveOpStore& store, const GraphHandle& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

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
        if (auto share = dim.tryAs<ShareOp::Input>(); share) {
            // This may be a share across tensors.
            if (share->getOp()->output.type() == DimensionType::Merge) {
                // Yes, it is.
                ++CountSuccessfulGenerations;
                res.emplace_back(store.get<ExpandOp>(dim));
            }
        } else {
            // This is just a simple repeat.
            ++CountSuccessfulGenerations;
            res.emplace_back(store.get<ExpandOp>(dim));
        }
    }
    CountDisallowedAttempts += interface.getDimensions().size() - countPlausible;
    return res;
}

} // namespace kas
