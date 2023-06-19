#include <algorithm>

#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Unfold.hpp"


namespace kas {

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

ColoredInterface UnfoldOp::applyToInterface(const ColoredInterface& interface) const {
    // Absorb dataDiscarding flag in outputRhs.
    return interface.substitute2to1(outputLhs, outputRhs, getInput(), true);
}

std::vector<const UnfoldOp *> UnfoldOp::Generate(PrimitiveOpStore& store, const ColoredInterface& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    // In addition, canonicalization can require that UnfoldOp chain be structured in ascending order of kernel size. This changes semantics but it seems to be fine.
    using enum DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallowsL { ShareR };
    std::vector<DimensionTypeWithOrder> disallowsR { ShareR };
    if (options.disallowUnfoldLAboveSplit) disallowsL.push_back(Split);
    if (options.disallowUnfoldLAboveShift) disallowsL.push_back(Shift);
    auto plausibleL = interface.filterOut(disallowsL);
    auto plausibleR = interface.filterOut(disallowsR);

    std::vector<const UnfoldOp *> result;
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
            // First check whether the kernel is small enough.
            // Absolute size.
            if (dimR.size().upperBoundEst(options.ctx) > options.maxUnfoldKernelSize) {
                ++CountKernelAbsolutelyTooLarge;
                continue;
            }
            // Relative size.
            auto quotient = dimL.size() / dimR.size();
            if (quotient.lowerBoundEst(options.ctx) < options.minimumRatio) {
                ++CountKernelRelativelyTooLarge;
                continue;
            }
            // Canonicalize unfold chains, requiring that UnfoldOp's with smaller kernels be first built.
            if (options.canonicalizeUnfoldOrder) {
                if (auto nextUnfold = dimL.tryAs<UnfoldOp::Input>(); nextUnfold) {
                    auto quotient = dimR.size() / nextUnfold->getOp()->outputRhs.size();
                    if (quotient.lowerBoundEst(options.ctx) < 1) {
                        ++CountCanonicalizedUnfoldChains;
                        continue;
                    }
                }
            }
            // Maybe we should rule out even sized kernels? TODO.
            ++CountSuccessfulGenerations;
            result.emplace_back(store.get<UnfoldOp>(dimL, dimR));
        }
    }
    CountDisallowedAttempts += totalAttempts - countPlausible;
    return result;
}

} // namespace kas
