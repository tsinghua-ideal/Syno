#include <algorithm>

#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Stride.hpp"


namespace kas {

std::size_t StrideOp::initialHash() const noexcept {
    std::size_t h = static_cast<std::size_t>(Type);
    HashCombine(h, stride);
    return h;
}

StrideOp::Values StrideOp::value(const Values& known) const {
    // This is different from other Op's. Because we need to set the initial orientation if allUnoriented.
    if (known.allValued()) return known;
    if (known.allUnoriented()) {
        return {{ Direction::Up, std::monostate{} }};
    }
    auto& [input, output] = known.values;
    auto stride = ConstValueNode::Create(this->stride);
    if (input.isOrientedUp()) { // Note that we must have set this to Up. This is sanity check. If input is valued, output must have been set beforehand.
        if (auto outputV = output.tryValue(); outputV) {
            // Out value -> in value.
            return {{ outputV * stride, outputV }};
        } else if (output.isUnorientedOrOrientedUp()) {
            // OK, but nothing to do.
            return {{ Direction::Up, output }};
        }
    }
    KAS_CRITICAL("Conflicting values for StrideOp: input = {}, output = {}", input, output);
}

ColoredInterface StrideOp::applyToInterface(const ColoredInterface& interface) const {
    // Add dataDiscarding flag.
    return interface.substitute1to1(output, getInput(), true);
}

std::vector<const StrideOp *> StrideOp::Generate(DimensionStore& store, const ColoredInterface& interface, GenerateOptions options) {
    ++CountGenerateInvocations;

    using enum DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallows { Unfold, Stride };
    if (options.disallowStrideAboveSplit) disallows.push_back(Split);
    if (options.disallowStrideAboveMergeR) disallows.push_back(MergeR);
    auto plausible = interface.filterOut(disallows);

    std::vector<const StrideOp *> result;
    CountGenerateAttempts += interface.size();
    std::size_t countPlausible = 0;
    for (auto&& [dim, color]: plausible) {
        ++countPlausible;
        for (Size stride: dim.size().sampleDivisors(options.ctx)) {
            // Disallow too large strides.
            if ((dim.size() * stride).upperBoundEst(options.ctx) > options.maxStridedDimSize) {
                ++CountSizeTooLarge;
                continue;
            }
            ++CountSuccessfulGenerations;
            result.emplace_back(store.get<StrideOp>(dim, stride));
        }
    }
    CountDisallowedAttempts += interface.size() - countPlausible;
    return result;
}

} // namespace kas
