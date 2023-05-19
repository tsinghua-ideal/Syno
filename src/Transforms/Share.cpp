#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

ShareOp::Values ShareOp::value(const Values& known) const {
    if (known.canSkipDeduction()) return known;
    auto& [inputLhs, inputRhs, output] = known.values;
    // In the following 3 cases, value propagates from one branch to the others.
    if (auto v = output.tryValue(); v) {
        if (inputLhs.isUnorientedOrOrientedUp() && inputRhs.isUnorientedOrOrientedUp()) { // Check.
            return {{ v, v, v }};
        }
    } else if (auto v = inputLhs.tryValue(); v) {
        if (inputRhs.isUnorientedOrOrientedUp() && output.isUnorientedOrOrientedDown()) { // Check.
            return {{ v, v, v }};
        }
    } else if (auto v = inputRhs.tryValue(); v) {
        if (inputLhs.isUnorientedOrOrientedUp() && output.isUnorientedOrOrientedDown()) { // Check.
            return {{ v, v, v }};
        }
    }
    // In the following 3 cases, orientation propagates from one branch to the others.
    else if (output.isOrientedUp()) {
        if (inputLhs.isUnorientedOrOrientedUp() && inputRhs.isUnorientedOrOrientedUp()) { // Check.
            return {{ Direction::Up, Direction::Up, Direction::Up }};
        }
    } else if (inputLhs.isOrientedDown()) {
        if (inputRhs.isUnorientedOrOrientedUp() && output.isUnorientedOrOrientedDown()) { // Check.
            return {{ Direction::Down, Direction::Up, Direction::Down }};
        }
    } else if (inputRhs.isOrientedDown()) {
        if (inputLhs.isUnorientedOrOrientedUp() && output.isUnorientedOrOrientedDown()) { // Check.
            return {{ Direction::Up, Direction::Down, Direction::Down }};
        }
    }
    // Otherwise, conficts.
    KAS_CRITICAL("Conflicting values for ShareOp: inputLhs = {}, inputRhs = {}, output = {}", inputLhs, inputRhs, output);
}

std::pair<bool, CompactColor> ShareOp::transformColor(CompactColor fro1, CompactColor fro2) const {
    // Require empty intersection.
    return { !(fro1 & fro2), fro1 | fro2 };
}

ColoredInterface ShareOp::applyToInterface(const ColoredInterface& interface) const {
    // Add new constraints.
    return interface.substitute1to2(output, getInputL(), getInputR(), true);
}

bool ShareOp::IsSharedDimensionCanonical(const PrimitiveOp *op, const Graph& graph) {
    if (op->getType() == DimensionType::Share) {
        const ShareOp& self = *static_cast<const ShareOp *>(op);
        if (auto outputShared = self.output.tryAs<ShareOp::Input>(); outputShared) {
            KAS_ASSERT(outputShared->getOrder() == Order::Left, "We are only allowed to chain ShareOp's to the left! This requirement should have been enforced in ShareOp::Generate().");
        }
        return true;
    }

    // We need to check whether the output of op is shared, i.e., ShareOp::Input. If so, we need to ensure that the hash is ascending from right to left, which is the canonical form.
    const ShareOp::Input *shared = nullptr, *sharedAnother = nullptr;
    if (auto rOp = dynamic_cast<const RepeatLikeOp *>(op); rOp) {
        shared = rOp->output.tryAs<ShareOp::Input>();
    } else if (auto sOp = dynamic_cast<const SplitLikeOp *>(op); sOp) {
        shared = sOp->outputLhs.tryAs<ShareOp::Input>();
        sharedAnother = sOp->outputRhs.tryAs<ShareOp::Input>();
    } else if (auto sOp = dynamic_cast<const MergeLikeOp *>(op); sOp) {
        shared = sOp->output.tryAs<ShareOp::Input>();
    }

    std::size_t thisHash = op->opHash();
    // Whether it is canonical to make sharedOutputDim shared.
    auto canonicalToHaveThisOutputShared = [&](const ShareOp::Input *sharedOutputDim) {
        if (!sharedOutputDim) {
            // Not shared, OK.
            return true;
        }

        // 2 cases.
        auto dim = [&]() -> std::optional<Dimension> {
            if (sharedOutputDim->getOrder() == Order::Left) {
                // If this outputDim is inputLhs of a ShareOp, then it is the last Dimension to be shared in the ShareOp chain. So we only need to compare with the other side of Share. Moreover, we should assume that the Op above exists.
                return sharedOutputDim->getOther();
            } else {
                // If this outputDim is inputRhs of a SharedOp, we need to dig even one layer deeper to find the previous Op.
                const auto& nextLayer = sharedOutputDim->getOp()->output;
                if (auto nextShare = nextLayer.tryAs<ShareOp::Input>(); nextShare) {
                    // Here it is! And we need to assert its existence.
                    KAS_ASSERT(nextShare->getOrder() == Order::Left);
                    return nextShare->getOther();
                } else {
                    // Seems no ShareOp beneath. This is the first slot of the Share chain!
                    return std::nullopt;
                }
            }
        }();
        if (!dim) {
            // This is the first Op above the Share chain. OK.
            return true;
        }

        std::size_t hash = 0;
        // Obtain the hash of the op above.
        auto getHash = [&](const auto& vertex, auto&& fromBranch) {
            hash = vertex.op.opHash();
            return true;
        };
        // It is possible that there is no op above.
        bool success = graph.visitAlong(*dim, Direction::Up).match(
            getHash,
            getHash,
            getHash
        );
        if (success) {
            if (thisHash == hash) {
                KAS_WARNING("Hash collision {} detected during canonicalization!", thisHash);
            }
            return thisHash >= hash;
        } else {
            // The previous slot is empty! Not canonical!
            return false;
        }
    };
    return canonicalToHaveThisOutputShared(shared) && canonicalToHaveThisOutputShared(sharedAnother);
}

std::vector<const ShareOp *> ShareOp::Generate(DimensionStore& store, const ColoredInterface& interface, GenerateOptions options) {
    ++CountGenerateInvocations;

    // Canonicalization requires that ShareOp only appears above Merge.
    // Also, it can be built above ShareL, to enable "chained" Share.
    using enum DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallows { ShareR, Split, Unfold, Stride, Shift };
    auto plausible = interface.filterOut(disallows);

    Allowance allowance { Size::Product(interface.getShape()), options.ctx };
    std::vector<const ShareOp *> result;
    if (interface.size() < options.dimUpperBound) {
        CountGenerateAttempts += interface.size();
        std::size_t countPlausible = 0;
        for (auto&& dim: plausible) {
            ++countPlausible;
            if (!allowance.withinAllowance(dim.size())) {
                ++CountAllowanceExceeded;
                continue;
            }
            if (dim.dimension.is(DimensionType::Share)) {
                // We can assert that this is left, because we have filtered ShareR out!
                auto& self = dim.dimension.as<ShareOp::Input>();
                KAS_ASSERT(self.getOrder() == Order::Left);
            }
            if (dim.color.countTags() + 1 > options.maxColorTags()) {
                // Too many color tags.
                ++CountMaximumTensorsExceeded;
                continue;
            }
            ++CountSuccessfulGenerations;
            result.emplace_back(store.get<ShareOp>(dim));
        }
        CountDisallowedAttempts += interface.size() - countPlausible;
    }
    return result;
}

} // namespace kas
