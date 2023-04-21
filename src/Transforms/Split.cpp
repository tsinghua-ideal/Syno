#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

SplitOp::Values SplitOp::value(const Values &known) const {
    if (known.canSkipDeduction()) return known;
    auto& [input, outputLhs, outputRhs] = known.values;
    auto block = ConstValueNode::Create(this->outputRhs.size());
    if (auto outputLV = outputLhs.tryValue(), outputRV = outputRhs.tryValue(); outputLV && outputRV) {
        // Value propagation pattern #1.
        if (outputLV && outputRV && input.isUnorientedOrOrientedUp()) { // Check.
            // Output iterators determine the input iterator. Typical in forward pipeline.
            return {{ outputLV * block + outputRV, outputLV, outputRV }};
        }
    } else if (auto inputV = input.tryValue(); inputV) {
        // Value propagation pattern #2.
        if (outputLhs.isUnorientedOrOrientedDown() && outputRhs.isUnorientedOrOrientedDown()) { // Check.
            // Input iterator determines the two output iterators. Typical in backward pipeline.
            return {{ inputV, inputV / block, inputV % block }};
        }
    } else if (outputLhs.isValuedOrOrientedUp() || outputRhs.isValuedOrOrientedUp()) { // Note that the two cannot be both valued.
        // Orientation propagation pattern #1.
        if (input.isUnorientedOrOrientedUp()) { // Check.
            // Propagate orientation to the other side, because input will be determined by outputs.
            return {{ Direction::Up, outputLhs, outputRhs }};
        }
    } else if (input.isOrientedDown()) {
        // Orientation propagation pattern #2.
        if (outputLhs.isUnorientedOrOrientedDown() && outputRhs.isUnorientedOrOrientedDown()) { // Check.
            // Input iterator will determine the two output iterators.
            return {{ Direction::Down, Direction::Down, Direction::Down }};
        }
    }
    KAS_CRITICAL("Conflicting values for SplitOp: input = {}, outputLhs = {}, outputRhs = {}", input, outputLhs, outputRhs);
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
