#include <algorithm>
#include <functional>

#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Stride.hpp"


namespace kas {

StrideOp::StrideOp(const Dimension& output, const Size& stride):
    RepeatLikeOp { output },
    stride { stride },
    sz { this->output.size() * this->stride },
    input { this }
{
    // StrideOp discards data.
    color.setDataDiscarding(true);
}

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

std::vector<const StrideOp *> StrideOp::Generate(PrimitiveOpStore& store, const Dimensions& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    using enum DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallows { ShareR, Unfold, Stride };
    if (options.disallowStrideAboveSplit) disallows.push_back(Split);
    if (options.disallowStrideAboveMergeR) disallows.push_back(MergeR);
    std::vector<Dimension> plausible;
    for (const auto& p: interface.filterOut(disallows)) {
        plausible.emplace_back(p);
    }

    Allowance allowance { interface.getShape().totalSize(), options.ctx };

    std::vector<const StrideOp *> result;
    CountGenerateAttempts += interface.size();
    for (Size stride: allowance.enumerateSizes(options.ctx)) {
        for (auto&& dim: plausible) {
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
