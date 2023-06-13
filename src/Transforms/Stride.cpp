#include <algorithm>
#include <functional>

#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Stride.hpp"


namespace kas {

std::size_t StrideOp::initialHash() const noexcept {
    std::size_t h = std::hash<DimensionType>{}(Type);
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
    if (input.isUnorientedOrOrientedUp()) { // If input is valued, output must have been set beforehand.
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

std::vector<const StrideOp *> StrideOp::Generate(DimensionStore& store, const ColoredInterface& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    using enum DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallows { ShareR, Unfold, Stride };
    if (options.disallowStrideAboveSplit) disallows.push_back(Split);
    if (options.disallowStrideAboveMergeR) disallows.push_back(MergeR);
    std::vector<std::reference_wrapper<const ColoredDimension>> plausible;
    for (const auto& p: interface.filterOut(disallows)) {
        plausible.emplace_back(std::cref(p));
    }

    Allowance allowance { interface.getShape().totalSize(), options.ctx };

    std::vector<const StrideOp *> result;
    CountGenerateAttempts += interface.size();
    for (Size stride: allowance.enumerateSizes(options.ctx)) {
        for (auto&& p: plausible) {
            auto&& [dim, color] = p.get();
            // Disallow too large strides.
            if ((dim.size() * stride).upperBoundEst(options.ctx) > options.maxStridedDimSize) {
                ++CountSizeTooLarge;
                continue;
            }
            ++CountSuccessfulGenerations;
            result.emplace_back(store.get<StrideOp>(dim, stride));
        }
    }
    CountDisallowedAttempts += interface.size() - plausible.size();
    return result;
}

} // namespace kas
