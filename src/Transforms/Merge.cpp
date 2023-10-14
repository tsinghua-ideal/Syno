#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Transforms/Merge.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

const Size& MergeOp::Input::size() const {
    switch (order) {
    case Order::Left:
        return getDerivedOp<MergeOp>()->majorSize;
    case Order::Right:
        return getDerivedOp<MergeOp>()->minorSize;
    }
}

MergeOp::MergeOp(const Dimension& output, const Size& block):
    MergeLikeOp { output },
    minorSize { block },
    majorSize { this->output.size() / this->minorSize },
    inputLhs { this, Order::Left },
    inputRhs { this, Order::Right }
{}

std::size_t MergeOp::initialHash() const noexcept {
    std::size_t h = DimensionTypeHash(Type);
    HashCombine(h, minorSize);
    return h;
}

MergeOp::Values MergeOp::value(const Values& known) const {
    if (known.canSkipDeduction()) return known;
    auto& [inputLhs, inputRhs, output] = known.values;
    auto block = ConstValueNode::Create(this->minorSize);
    if (auto outputV = output.tryValue(); outputV) {
        // Value propagation pattern #1.
        if (inputLhs.isUnorientedOrOrientedUp() && inputRhs.isUnorientedOrOrientedUp()) { // Check.
            // Output iterator determines the two input iterators. Typical in forward pipeline.
            return {{ outputV / block, outputV % block, outputV }};
        }
    } else if (auto inputLV = inputLhs.tryValue(), inputRV = inputRhs.tryValue(); inputLV && inputRV) {
        // Value propagation pattern #2.
        if (inputLV && inputRV && output.isUnorientedOrOrientedDown()) { // Check.
            // Input iterators determine the output iterator. Typical in backward pipeline.
            return {{ inputLV, inputRV, inputLV * block + inputRV }};
        }
    } else if (output.isOrientedUp()) {
        // Orientation propagation pattern #1.
        if (inputLhs.isUnorientedOrOrientedUp() && inputRhs.isUnorientedOrOrientedUp()) { // Check.
            // Output iterator will determine the two input iterators.
            return {{ Direction::Up, Direction::Up, Direction::Up }};
        }
    } else if (inputLhs.isValuedOrOrientedDown() || inputRhs.isValuedOrOrientedDown()) { // Note that the two cannot be both valued.
        // Orientation propagation pattern #2.
        if (output.isUnorientedOrOrientedDown()) { // Check.
            // Propagate orientation to the other side, because output will be determined by inputs.
            return {{ inputLhs, inputRhs, Direction::Down }};
        }
    }
    // Otherwise, there must have been conflicts.
    KAS_CRITICAL("Conflicting values for MergeOp: inputLhs = {}, inputRhs = {}, output = {}", inputLhs, inputRhs, output);
}

std::vector<const MergeOp *> MergeOp::Generate(PrimitiveOpStore& store, const GraphHandle& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    // Canonicalization. Manually handle SplitOp, StrideOp(s<B) and UnfoldOp(k<B).
    using T = DimensionTypeWithOrder;
    auto plausible = interface.filterOut({ T::ShareL, T::ShareR, T::Split, T::MergeR });

    std::vector<const MergeOp *> res;
    auto checkThenAdd = [&options, &store, &res](const Dimension& dim, Size&& block) {
        if (auto split = dim.tryAs<SplitOp::Input>(); split) {
            if (split->getOp()->outputRhs.size() == block) {
                ++CountConteractedSplits;
                return; // This is pointless!
            }
        }
        if (auto r = dim.tryAs<Reduce>(); r) {
            ++CountUselessImmediateReductions;
            return; // For identity-mapped, sum-reduced, no need for this! TODO: if more types are added, change this.
        }
        if (options.disallowMergeWithLargeBlockAboveStride) {
            if (auto s = dim.tryAs<StrideOp::Input>(); s) {
                if ((block / s->getDerivedOp<StrideOp>()->getStride()).lowerBoundEst(options.ctx) > static_cast<std::size_t>(1)) {
                    ++CountDisallowedAboveStride;
                    return;
                }
            }
        }
        if (options.disallowMergeWithLargeBlockAboveUnfold) {
            if (auto u = dim.tryAs<UnfoldOp::Input>(); u) {
                if ((block / u->getOp()->outputRhs.size()).lowerBoundEst(options.ctx) > static_cast<std::size_t>(1)) {
                    ++CountDisallowedAboveUnfold;
                    return;
                }
            }
        }
        ++CountSuccessfulGenerations;
        res.emplace_back(store.get<MergeOp>(dim, std::move(block)));
    };
    CountGenerateAttempts += interface.getDimensions().size();
    std::size_t countPlausible = 0;
    for (auto&& dim: plausible) {
        ++countPlausible;
        for (Size sizeR: options.allowance.enumerateDivisors(dim.size())) {
            checkThenAdd(dim, std::move(sizeR));
        }
    }
    CountDisallowedAttempts += interface.getDimensions().size() - countPlausible;
    return res;
}

} // namespace kas
