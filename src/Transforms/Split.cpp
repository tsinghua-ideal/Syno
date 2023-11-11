#include "KAS/Core/Graph.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Transforms/OperationStore.hpp"
#include "KAS/Transforms/Reshape.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

bool SplitOp::isEqual(const Operation& other) const {
    const SplitOp& rhs = static_cast<const SplitOp&>(other);
    return outputLhs == rhs.outputLhs && outputRhs == rhs.outputRhs;
}

SplitOp::SplitOp(const Dimension& outputLhs, const Dimension& outputRhs):
    SplitLikeOp { outputLhs, outputRhs },
    sz { this->outputLhs.size() * this->outputRhs.size() },
    input { this }
{}

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

std::vector<const SplitOp *> SplitOp::Generate(OperationStore& store, const Topmost& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    const Graph& graph = options.graph;

    // Canonicalization requires SplitOp to be chained.
    using T = DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallowsL { T::ShareR, T::Split };
    if (options.disallowSplitLAboveUnfold) disallowsL.push_back(T::Unfold);
    std::vector<DimensionTypeWithOrder> disallowsR { T::ShareR };
    if (options.disallowSplitRAboveUnfold) disallowsR.push_back(T::Unfold);
    if (options.disallowSplitRAboveStride) disallowsR.push_back(T::Stride);
    auto plausibleL = interface.filterOut(std::move(disallowsL));
    auto plausibleR = interface.filterOut(std::move(disallowsR));

    ReshapeCanonicalizer canonicalizer;
    graph.accept(canonicalizer);

    std::vector<const SplitOp *> result;
    auto checkThenAdd = [&store, &canonicalizer, &result, &ctx = options.ctx, &graph, &old = options.couldHaveBeenDoneBeforeLastContractionStage](const Dimension& dimL, const Dimension& dimR) {
        if (old.contains(dimL) && old.contains(dimR)) {
            ++CountCouldHaveBeenDoneBeforeLastContractionStage;
            return;
        }
        // Perform canonicalization for reshape.
        if (canonicalizer.at(dimL).isAdjacentTo(canonicalizer.at(dimR))) {
            // They are redundant!
            // Because the split dimensions are merged or reduced.
            ++CountCounteractedMergesAndReduces;
            return;
        }
        // Check if we can move this Split down the Share.
        if (
            auto lShare = dimL.tryAs<ShareOp::Input>(), rShare = dimR.tryAs<ShareOp::Input>();
            lShare && rShare && lShare->getDerivedOp<ShareOp>()->getRhsOrigin() == rShare->getDerivedOp<ShareOp>()->getRhsOrigin()
        ) {
            ++CountCanBeDeferredAfterContraction;
            return;
        }
        // Check orderedness.
        if (
            const auto& colorL = graph.colorOf(dimL), & colorR = graph.colorOf(dimR);
            // Same unorderedness.
            colorL.isUnordered() && colorR.isUnordered() && colorL.getUnorderedScope() == colorR.getUnorderedScope()
        ) {
            // They are from the same unordered dim.
            // No matter whether they are adjacent, it is sure that if they are from the same merge block, they are redundant.
            auto mergeL = dimL, mergeR = dimR;
            while (auto m = mergeL.tryAs<MergeOp::Input>()) mergeL = m->getOp()->output;
            while (auto m = mergeR.tryAs<MergeOp::Input>()) mergeR = m->getOp()->output;
            if (mergeL == mergeR) {
                ++CountCounteractedUnorderedMerges;
                return;
            }
        }
        // Check that the created size is valid.
        if (auto product = dimL.size() * dimR.size(); !ctx.isSizeValid(product)) {
            ++CountInvalidProductSize;
            return;
        }
        ++CountSuccessfulGenerations;
        result.emplace_back(store.get<SplitOp>(dimL, dimR));
    };
    const auto totalAttempts = interface.getDimensions().size() * (interface.getDimensions().size() - 1);
    CountGenerateAttempts += totalAttempts;
    std::size_t countPlausible = 0;
    for (auto&& dimL: plausibleL) {
        for (auto&& dimR: plausibleR) {
            if (dimL == dimR) continue;
            ++countPlausible;
            checkThenAdd(dimL, dimR);
        }
    }
    CountDisallowedAttempts += totalAttempts - countPlausible;
    return result;
}

} // namespace kas
