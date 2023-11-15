#include <algorithm>

#include "KAS/Transforms/OperationStore.hpp"
#include "KAS/Transforms/Unfold.hpp"


namespace kas {

bool UnfoldOp::isEqual(const Operation& other) const {
    const UnfoldOp& rhs = static_cast<const UnfoldOp&>(other);
    return outputLhs == rhs.outputLhs && outputRhs == rhs.outputRhs;
}

Color UnfoldOp::Input::computeColor(const GraphBuilder& graphBuilder) const {
    // Absorb dataDiscardingFlag in outputRhs.
    return SplitLikeOp::Input::computeColor(graphBuilder).setDataDiscarding(graphBuilder.colorOf(op->outputLhs).isDataDiscarding());
}

UnfoldOp::UnfoldOp(const Dimension& outputLhs, const Dimension& outputRhs):
    SplitLikeOp { outputLhs, outputRhs },
    input { this }
{}

UnfoldOp::Values UnfoldOp::value(const Values& known) const {
    if (known.canSkipDeduction()) return known;
    auto& [input, outputLhs, outputRhs] = known.values;
    auto originalSize = this->outputLhs.size();
    auto kernelSize = this->outputRhs.size();
    auto outOfBoundFraction = kernelSize / originalSize;
    auto kernel = ConstValueNode::Create(kernelSize);
    auto halfKernel = kernel / ImmediateValueNode::Two;
    if (auto outputLV = outputLhs.tryValue(), outputRV = outputRhs.tryValue(); outputLV && outputRV) {
        // Major output and minor output determine input. Typical in forward pipeline.
        if (input.isUnorientedOrOrientedUp()) { // Check.
            return {{ IntervalBoundValueNode::Create(outputLV + outputRV - halfKernel, originalSize, outOfBoundFraction), outputLV, outputRV }};
        }
    } else if (auto inputV = input.tryValue(), outputRV = outputRhs.tryValue(); inputV && outputRV) {
        // Input and minor output determine major output. This can convert scatter patttern to gather pattern. Typical in backward pipeline.
        if (outputLhs.isUnorientedOrOrientedDown()) { // Check.
            return {{ inputV, IntervalBoundValueNode::Create(inputV - outputRV + halfKernel, originalSize, outOfBoundFraction), outputRV }};
        }
    } else if (!outputRhs.isOrientedDown()) { // Only Direction::Down is illegal!
        if (input.isValuedOrOrientedDown()) {
            // input -> outputLhs.
            if (outputLhs.isUnorientedOrOrientedDown()) {
                return {{ input, Direction::Down, outputRhs }};
            }
        } else if (outputLhs.isValuedOrOrientedUp()) {
            // outputLhs -> input.
            if (input.isUnorientedOrOrientedUp()) {
                return {{ Direction::Up, outputLhs, outputRhs }};
            }
        } else if (outputLhs.isUnoriented() && input.isUnoriented()) {
            // Nothing to deduce.
            return {{ std::monostate{}, std::monostate{}, outputRhs }};
        }
    }
    // Otherwise, conflict.
    KAS_CRITICAL("Conflicting values for UnfoldOp: input = {}, outputLhs = {}, outputRhs = {}", input, outputLhs, outputRhs);
}

std::vector<const UnfoldOp *> UnfoldOp::Generate(OperationStore& store, const Topmost& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    const Graph& graph = options.graph;

    // In addition, canonicalization can require that UnfoldOp chain be structured in ascending order of kernel size. This changes semantics but it seems to be fine.
    using enum DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallowsL { ShareR };
    std::vector<DimensionTypeWithOrder> disallowsR { Reduce, ShareR };
    if (options.disallowUnfoldLAboveSplit) disallowsL.push_back(Split);
    if (options.disallowUnfoldLAboveShift) disallowsL.push_back(Shift);
    if (options.disallowUnfoldLAboveMergeR) disallowsL.push_back(MergeR);
    auto plausibleL = interface.filterOut(std::move(disallowsL));
    auto plausibleR = interface.filterOut(std::move(disallowsR));

    std::vector<const UnfoldOp *> result;
    const auto totalAttempts = interface.getDimensions().size() * (interface.getDimensions().size() - 1);
    CountGenerateAttempts += totalAttempts;
    std::size_t countPlausible = 0;
    for (auto&& dimL: plausibleL) {
        for (auto&& dimR: plausibleR) {
            if (dimL == dimR) continue;
            ++countPlausible;
            if (graph.colorOf(dimL).endsUpReduce()) {
                ++CountDoubleReduction;
                continue;
            }
            const Size& kernelSize = dimR.size();
            // Optionally require that the kernel size is odd.
            // A precondition is that we are sure that all the fractions evaluate to integers.
            if (
                options.requiresOddKernelSizeInUnfold &&
                std::ranges::any_of(
                    kernelSize.evalAllConsts<std::size_t>(options.ctx),
                    [](auto x) { return x % 2 == 0; }
                )
            ) {
                ++CountEvenKernelSize;
                continue;
            }
            // First check whether the kernel is small enough.
            // Absolute size.
            if (kernelSize.upperBoundEst(options.ctx) > options.maxUnfoldKernelSize) {
                ++CountKernelAbsolutelyTooLarge;
                continue;
            }
            // Relative size.
            auto quotient = dimL.size() / kernelSize;
            if (boost::rational_cast<float>(quotient.lowerBoundEst(options.ctx)) < options.minimumRatio) {
                ++CountKernelRelativelyTooLarge;
                continue;
            }
            // Canonicalize unfold chains, requiring that UnfoldOp's with smaller kernels be first built.
            if (options.canonicalizeUnfoldOrder) {
                if (auto nextUnfold = dimL.tryAs<UnfoldOp::Input>(); nextUnfold) {
                    auto quotient = kernelSize / nextUnfold->getDerivedOp<UnfoldOp>()->getWindow();
                    if (quotient.lowerBoundEst(options.ctx) < 1_uz) {
                        ++CountCanonicalizedUnfoldChains;
                        continue;
                    }
                }
            }
            ++CountSuccessfulGenerations;
            result.emplace_back(store.get<UnfoldOp>(dimL, dimR));
        }
    }
    CountDisallowedAttempts += totalAttempts - countPlausible;
    return result;
}

} // namespace kas
