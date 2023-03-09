#include <algorithm>

#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Unfold.hpp"


namespace kas {

IteratorValue UnfoldOp::value(const IteratorValue& outputMajor, const IteratorValue& outputMinor) const {
    auto original = ConstValueNode::Create(outputLhs.size());
    auto kernel = ConstValueNode::Create(outputRhs.size());
    auto access = outputMajor + outputMinor - (kernel - ImmediateValueNode::One) / ImmediateValueNode::Two;
    return IntervalBoundValueNode::Create(access, ImmediateValueNode::Zero, original);
}

std::size_t UnfoldOp::CountColorTrials = 0;
std::size_t UnfoldOp::CountColorSuccesses = 0;
bool UnfoldOp::transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const {
    ++CountColorTrials;
    auto& outLhs = interface[outputLhs];
    auto& outRhs = interface[outputRhs];
    // Unfold creates clear dimensions.
    if (outRhs.isSingle()) { // So we must not violate existing constraints.
        return false;
    }
    if (outputRhs.size().isGeneral()) { // [Single Statement] We know that general dimension cannot be clear.
        return false;
    }
    // The `substitute` removes outputRhs, so actually no need to make it clear.
    colors.assign(interface, outputRhs, Colors::Clear);
    // Unfold preserves colors in the major dimension.
    colors.substitute(interface, outputLhs, outputRhs, { getInput(), outLhs.color });
    colors.simplify(interface);
    ++CountColorSuccesses;
    return true;
}

std::vector<const UnfoldOp *> UnfoldOp::Generate(DimensionStore& store, const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options) {
    std::vector<const UnfoldOp *> result;
    if (outputShape.size() > options.dimLowerBound) {
        std::vector<std::size_t> generals;
        std::vector<std::size_t> windows;
        auto meta = options.ctx.getCoefficientMetadata();
        auto coefficientCount = options.ctx.getCoefficientCount();
        for (std::size_t i = 0; i < outputShape.size(); ++i) {
            const auto& size = outputShape[i].size();
            auto p = size.getPrimary();
            if (std::ranges::any_of(p, [](auto i) { return i != 0; })) {
                // Here, we only allow an axis with primary variable to be unfolded. TODO: relax this?
                generals.emplace_back(i);
            } else {
                auto coefficient = size.getCoefficient();
                bool isEven = false;
                for (std::size_t i = 0; i < coefficientCount; ++i) {
                    if (coefficient[i] != 0 && !meta[i].isOdd) {
                        isEven = true;
                    }
                }
                if (!isEven) {
                    windows.emplace_back(i);
                }
            }
        }
        for (auto general: generals) {
            for (auto window: windows) {
                KAS_ASSERT(general != window);
                result.emplace_back(store.get<UnfoldOp>(outputShape[general], outputShape[window]));
            }
        }
    }
    return result;
}

} // namespace kas
