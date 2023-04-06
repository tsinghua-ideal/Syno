#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Common.hpp"


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
    if (!input && !outputLhs && !outputRhs) {
        return { .input = 0, .outputLhs = 0, .outputRhs = 0 };
    } else if (!input && outputLhs && !outputRhs) {
        return { .input = 0, .outputLhs = -1, .outputRhs = 1 };
    } else if (!input && !outputLhs && outputRhs) {
        return { .input = 0, .outputLhs = 1, .outputRhs = -1 };
    } else {
        KAS_UNREACHABLE("Not possible to call ordering() on SplitOp with input = {}, outputLhs = {}, outputRhs = {}", input.hasValue(), outputLhs.hasValue(), outputRhs.hasValue());
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
    auto checkNotMergeThenAdd = [&store, &result](const Dimension& dimL, const Dimension& dimR) {
        if (auto l = dimL.tryAs<MergeOp::Input>(); l) {
            if (auto r = dimR.tryAs<MergeOp::Input>(); r) {
                if (l->getOp() == r->getOp()) {
                    return; // They are just the same merge!
                }
            }
        }
        result.emplace_back(store.get<SplitOp>(dimL, dimR));
    };
    if (outputShape.size() > options.dimLowerBound) {
        for (std::size_t i = 0; i < outputShape.size(); ++i) {
            for (std::size_t j = 0; j < outputShape.size(); ++j) {
                if (i == j) continue;
                checkNotMergeThenAdd(outputShape[i], outputShape[j]);
            }
        }
    }
    return result;
}

} // namespace kas
