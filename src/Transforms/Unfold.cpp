#include <algorithm>

#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Unfold.hpp"


namespace kas {

std::size_t UnfoldOp::Input::hash() const noexcept {
    auto h = static_cast<std::size_t>(type());
    HashCombine(h, op->outputLhs.hash());
    HashCombine(h, op->outputRhs.hash());
    return h;
}

IteratorValue UnfoldOp::value(const IteratorValue& outputMajor, const IteratorValue& outputMinor) const {
    auto original = ConstValueNode::Create(outputLhs.size());
    auto kernel = ConstValueNode::Create(outputRhs.size());
    auto access = outputMajor + outputMinor - (kernel - ImmediateValueNode::One) / ImmediateValueNode::Two;
    return IntervalBoundValueNode::Create(access, ImmediateValueNode::Zero, original);
}

std::vector<std::unique_ptr<UnfoldOp>> UnfoldOp::Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options) {
    std::vector<std::unique_ptr<UnfoldOp>> result;
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
