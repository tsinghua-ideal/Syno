#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Split.hpp"


namespace kas {

IteratorValue SplitOp::value(const IteratorValue &outputMajor, const IteratorValue &outputMinor) const {
    auto block = ConstValueNode::Create(outputRhs.size());
    return outputMajor * block + outputMinor;
}

bool SplitOp::transformColors(ColoredInterface& interface, Colors& colors, Colors::Options options) const {
    auto outLhs = interface.binarySearch(outputLhs);
    auto outRhs = interface.binarySearch(outputRhs);
    if (outLhs->isUnknown() && outRhs->isUnknown()) { // Pass around the Unknown.
        colors.substitute(interface, outputLhs, outputRhs, { getInput(), Colors::Unknown });
    } else if (!outLhs->isUnknown() && !outRhs->isUnknown()) {
        if (outLhs->color != outRhs->color) {
            return false; // Because Split preserves colors.
        }
        colors.substitute(interface, outputLhs, outputRhs, { getInput(), outLhs->color });
    } else {
        auto& known = outLhs->isUnknown() ? outRhs : outLhs;
        auto& unknown = outLhs->isUnknown() ? outLhs : outRhs;
        colors.assign(interface, unknown->dimension, known->color); // We have solved the unknown color.
        colors.substitute(interface, outputLhs, outputRhs, { getInput(), known->color });
    }
    colors.simplify(interface);
    return true;
}

std::vector<const SplitOp *> SplitOp::Generate(DimensionStore& store, const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options) {
    std::vector<const SplitOp *> result;
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
