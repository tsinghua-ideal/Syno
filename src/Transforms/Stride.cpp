#include <algorithm>
#include <functional>

#include "KAS/Transforms/OperationStore.hpp"
#include "KAS/Transforms/Stride.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

bool StrideOp::isEqual(const Operation& other) const {
    const StrideOp& rhs = static_cast<const StrideOp&>(other);
    return output == rhs.output && stride == rhs.stride;
}

Color StrideOp::Input::computeColor(const GraphBuilder &graphBuilder) const {
    // StrideOp discards data.
    return RepeatLikeOp::Input::computeColor(graphBuilder).setDataDiscarding(true);
}

StrideOp::StrideOp(const Dimension& output, const Size& stride):
    RepeatLikeOp { output },
    stride { stride },
    sz { this->output.size() * this->stride },
    input { this }
{}

std::size_t StrideOp::initialHash() const noexcept {
    std::size_t h = DimensionTypeHash(Type);
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

std::vector<const StrideOp *> StrideOp::Generate(OperationStore& store, const Topmost& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    using enum DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallows { ShareR, Unfold, Stride };
    if (options.disallowStrideAboveSplit) disallows.push_back(Split);
    if (options.disallowStrideAboveMergeR) disallows.push_back(MergeR);
    auto plausible = ranges::to<std::vector<Dimension>>(interface.filterOut(std::move(disallows)));

    std::vector<const StrideOp *> result;
    CountGenerateAttempts += interface.getDimensions().size();
    for (Size stride: options.allowance.enumerateSizes()) {
        for (auto&& dim: plausible) {
            auto product = dim.size() * stride;
            // Disallow too large strides.
            if (product.upperBoundEst(options.ctx) > options.maxStridedDimSize) {
                ++CountSizeTooLarge;
                continue;
            }
            // Check if product is valid.
            if (!options.ctx.isSizeValid(product)) {
                ++CountInvalidProductSize;
                continue;
            }
            ++CountSuccessfulGenerations;
            result.emplace_back(store.get<StrideOp>(dim, stride));
        }
    }
    CountDisallowedAttempts += interface.getDimensions().size() - plausible.size();
    return result;
}

} // namespace kas
