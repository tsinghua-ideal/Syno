#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Split.hpp"


namespace kas {

std::size_t SplitOp::Input::hash() const noexcept {
    auto h = static_cast<std::size_t>(type());
    HashCombine(h, op->outputLhs.hash());
    HashCombine(h, op->outputRhs.hash());
    return h;
}

IteratorValue SplitOp::value(const IteratorValue &outputMajor, const IteratorValue &outputMinor) const {
    auto block = ConstValueNode::Create(outputRhs.size());
    return outputMajor * block + outputMinor;
}

std::vector<std::unique_ptr<SplitOp>> SplitOp::Generate(DimensionStore& store, const Interface& outputShape, GenerateOptions options) {
    std::vector<std::unique_ptr<SplitOp>> result;
    if (outputShape.size() > options.dimLowerBound) {
        for (std::size_t i = 0; i < outputShape.size(); ++i) {
            for (std::size_t j = 0; j < outputShape.size(); ++j) {
                if (i == j) continue;
                // Merged to the dimension at front.
                result.emplace_back(store.get<SplitOp>(outputShape[i], outputShape[j]));
            }
        }
    }
    return result;
}

} // namespace kas
