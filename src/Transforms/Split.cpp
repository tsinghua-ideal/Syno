#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Split.hpp"


namespace kas {

SplitOp::IteratorValues SplitOp::value(const IteratorValues &known) const {
    auto& [input, outputLhs, outputRhs] = known;
    auto block = ConstValueNode::Create(this->outputRhs.size());
    if (!input && outputLhs && outputRhs) { // Output to input.
        return {{ .input = outputLhs * block + outputRhs }};
    } else if (input && !outputLhs && !outputRhs) { // Input to output.
        return {{ .outputLhs = input / block, .outputRhs = input % block }};
    } else if (input && outputLhs.hasValue() != outputRhs.hasValue()) { // Hard fail.
        KAS_CRITICAL("Conflicting values for SplitOp: input = {}, outputLhs = {}, outputRhs = {}", input.hasValue(), outputLhs.hasValue(), outputRhs.hasValue());
    } else { // Soft fail.
        return {};
    }
}

SplitOp::OrderingValues SplitOp::ordering(const IteratorValues &known) const {
    auto& [input, outputLhs, outputRhs] = known;
    if (!input && outputLhs && !outputRhs) {
        return { .input = 0, .outputLhs = -1, .outputRhs = 1 };
    } else if (!input && !outputLhs && outputRhs) {
        return { .input = 0, .outputLhs = 1, .outputRhs = -1 };
    } else {
        return { .input = -1, .outputLhs = -1, .outputRhs = -1 };
    }
}

std::size_t SplitOp::CountColorTrials = 0;
std::size_t SplitOp::CountColorSuccesses = 0;
bool SplitOp::transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const {
    ++CountColorTrials;
    auto& outLhs = interface[outputLhs];
    auto& outRhs = interface[outputRhs];
    if (outLhs.isUnknown() && outRhs.isUnknown()) { // Pass around the Unknown.
        colors.substitute(interface, outputLhs, outputRhs, { getInput(), Colors::Unknown });
    } else if (!outLhs.isUnknown() && !outRhs.isUnknown()) {
        if (outLhs.color != outRhs.color) {
            return false; // Because Split preserves colors.
        }
        colors.substitute(interface, outputLhs, outputRhs, { getInput(), outLhs.color });
    } else {
        auto& known = outLhs.isUnknown() ? outRhs : outLhs;
        auto& unknown = outLhs.isUnknown() ? outLhs : outRhs;
        // We have solved the unknown color.
        // `substitute` removes unknown, so actually no need to call assign here.
        colors.assign(interface, unknown.dimension, known.color);
        colors.substitute(interface, outputLhs, outputRhs, { getInput(), known.color });
    }
    colors.simplify(interface);
    ++CountColorSuccesses;
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
