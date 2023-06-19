#include "KAS/Core/MapReduce.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

SplitOp::Values SplitOp::value(const Values &known) const {
    if (known.canSkipDeduction()) return known;
    auto& [input, outputLhs, outputRhs] = known.values;
    auto block = ConstValueNode::Create(this->outputRhs.size());
    if (auto outputLV = outputLhs.tryValue(), outputRV = outputRhs.tryValue(); outputLV && outputRV) {
        // Value propagation pattern #1.
        if (outputLV && outputRV && input.isUnorientedOrOrientedUp()) { // Check.
            // Output iterators determine the input iterator. Typical in forward pipeline.
            return {{ outputLV * block + outputRV, outputLV, outputRV }};
        }
    } else if (auto inputV = input.tryValue(); inputV) {
        // Value propagation pattern #2.
        if (outputLhs.isUnorientedOrOrientedDown() && outputRhs.isUnorientedOrOrientedDown()) { // Check.
            // Input iterator determines the two output iterators. Typical in backward pipeline.
            return {{ inputV, inputV / block, inputV % block }};
        }
    } else if (outputLhs.isValuedOrOrientedUp() || outputRhs.isValuedOrOrientedUp()) { // Note that the two cannot be both valued.
        // Orientation propagation pattern #1.
        if (input.isUnorientedOrOrientedUp()) { // Check.
            // Propagate orientation to the other side, because input will be determined by outputs.
            return {{ Direction::Up, outputLhs, outputRhs }};
        }
    } else if (input.isOrientedDown()) {
        // Orientation propagation pattern #2.
        if (outputLhs.isUnorientedOrOrientedDown() && outputRhs.isUnorientedOrOrientedDown()) { // Check.
            // Input iterator will determine the two output iterators.
            return {{ Direction::Down, Direction::Down, Direction::Down }};
        }
    }
    KAS_CRITICAL("Conflicting values for SplitOp: input = {}, outputLhs = {}, outputRhs = {}", input, outputLhs, outputRhs);
}

std::vector<const SplitOp *> SplitOp::Generate(PrimitiveOpStore& store, const ColoredInterface& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    // Canonicalization requires SplitOp to be chained.
    using T = DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallowsL { T::ShareR, T::Split, T::Unfold };
    std::vector<DimensionTypeWithOrder> disallowsR { T::ShareR };
    if (options.disallowSplitRAboveUnfold) disallowsR.push_back(T::Unfold);
    if (options.disallowSplitRAboveStride) disallowsR.push_back(T::Stride);
    auto plausibleL = interface.filterOut(disallowsL);
    auto plausibleR = interface.filterOut(disallowsR);

    std::vector<const SplitOp *> result;
    auto checkThenAdd = [&store, &result, disallowDiscontinuousView = options.disallowDiscontinuousView](const Dimension& dimL, const Dimension& dimR) {
        if (auto l = dimL.tryAs<MergeOp::Input>(); l) {
            if (auto r = dimR.tryAs<MergeOp::Input>(); r) {
                if (l->getOp() == r->getOp()) {
                    if (l->getOrder() == Order::Left && r->getOrder() == Order::Right) {
                        // They are just the same merge!
                        ++CountCounteractedMerges;
                        return;
                    } else if (disallowDiscontinuousView && l->getOrder() == Order::Right && r->getOrder() == Order::Left) {
                        // This is a discontinuous view.
                        ++CountDisallowedDiscontinuousViews;
                        return;
                    }
                }
            }
        }
        if (auto l = dimL.tryAs<MapReduce>(); l) {
            if (auto r = dimR.tryAs<MapReduce>(); r) {
                // For identity-mapped, sum-reduced, no need for this! TODO: if more types are added, change this.
                ++CountUselessImmediateReductions;
                return;
            }
        }
        ++CountSuccessfulGenerations;
        result.emplace_back(store.get<SplitOp>(dimL, dimR));
    };
    const auto totalAttempts = interface.size() * interface.size() - interface.size();
    CountGenerateAttempts += totalAttempts;
    std::size_t countPlausible = 0;
    for (auto&& [dimL, colorL]: plausibleL) {
        for (auto&& [dimR, colorR]: plausibleR) {
            if (dimL == dimR) continue;
            ++countPlausible;
            if (!colorL.disjoint(colorR)) {
                ++CountConflictingColors;
                continue;
            }
            checkThenAdd(dimL, dimR);
        }
    }
    CountDisallowedAttempts += totalAttempts - countPlausible;
    return result;
}

} // namespace kas
