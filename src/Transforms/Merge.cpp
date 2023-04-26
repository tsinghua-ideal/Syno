#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Transforms/Merge.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

const Size& MergeOp::Input::size() const noexcept {
    switch (order) {
    case Order::Left:
        return getDerivedOp<MergeOp>()->majorSize;
    case Order::Right:
        return getDerivedOp<MergeOp>()->minorSize;
    }
}

std::size_t MergeOp::initialHash() const noexcept {
    std::size_t h = static_cast<std::size_t>(Type);
    HashCombine(h, minorSize);
    return h;
}

MergeOp::Values MergeOp::value(const Values& known) const {
    if (known.canSkipDeduction()) return known;
    auto& [inputLhs, inputRhs, output] = known.values;
    auto block = ConstValueNode::Create(this->minorSize);
    if (auto outputV = output.tryValue(); outputV) {
        // Value propagation pattern #1.
        if (inputLhs.isUnorientedOrOrientedUp() && inputRhs.isUnorientedOrOrientedUp()) { // Check.
            // Output iterator determines the two input iterators. Typical in forward pipeline.
            return {{ outputV / block, outputV % block, outputV }};
        }
    } else if (auto inputLV = inputLhs.tryValue(), inputRV = inputRhs.tryValue(); inputLV && inputRV) {
        // Value propagation pattern #2.
        if (inputLV && inputRV && output.isUnorientedOrOrientedDown()) { // Check.
            // Input iterators determine the output iterator. Typical in backward pipeline.
            return {{ inputLV, inputRV, inputLV * block + inputRV }};
        }
    } else if (output.isOrientedUp()) {
        // Orientation propagation pattern #1.
        if (inputLhs.isUnorientedOrOrientedUp() && inputRhs.isUnorientedOrOrientedUp()) { // Check.
            // Output iterator will determine the two input iterators.
            return {{ Direction::Up, Direction::Up, Direction::Up }};
        }
    } else if (inputLhs.isValuedOrOrientedDown() || inputRhs.isValuedOrOrientedDown()) { // Note that the two cannot be both valued.
        // Orientation propagation pattern #2.
        if (output.isUnorientedOrOrientedDown()) { // Check.
            // Propagate orientation to the other side, because output will be determined by inputs.
            return {{ inputLhs, inputRhs, Direction::Down }};
        }
    }
    // Otherwise, there must have been conflicts.
    KAS_CRITICAL("Conflicting values for MergeOp: inputLhs = {}, inputRhs = {}, output = {}", inputLhs, inputRhs, output);
}

std::size_t MergeOp::CountColorTrials = 0;
std::size_t MergeOp::CountColorSuccesses = 0;
bool MergeOp::transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const {
    ++CountColorTrials;
    auto& out = interface[output];
    colors.substitute(interface, output, { getInputL(), out.color }, { getInputR(), out.color });
    colors.simplify(interface); // Actually not needed.
    ++CountColorSuccesses;
    return true;
}

std::vector<const MergeOp *> MergeOp::Generate(DimensionStore& store, const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options) {
    const auto& ctx = options.ctx;
    auto primaryCount = ctx.getPrimaryCount(), coefficientCount = ctx.getCoefficientCount();

    std::vector<const MergeOp *> res;
    auto checkThenAdd = [&ctx, &store, &res](const Dimension& dim, auto&& block) {
        if (auto split = dim.tryAs<SplitOp::Input>(); split) {
            if (split->getOp()->outputRhs.size() == block) {
                return; // This is pointless!
            } else if (split->getOp()->outputLhs.size() == block) {
                return; // Maybe this is not pointless (view it in different shape?), but ban it for now.
            }
        }
        if (auto r = dim.tryAs<MapReduceOp>(); r) {
            return; // For identity-mapped, sum-reduced, no need for this! TODO: if more types are added, change this.
        }
        // Check that the sizes are realistic.
        if (block.isRealistic(ctx) && (dim.size() / block).isRealistic(ctx)) {
            res.emplace_back(store.get<MergeOp>(dim, std::forward<decltype(block)>(block)));
        }
    };
    if (outputShape.size() < options.dimUpperBound) {
        for (std::size_t i = 0; i < outputShape.size(); ++i) {
            const auto& size = outputShape[i].size();
            if (size.getPrimaryPowersSum() == 0) {
                // Here we only split dimensions that have >= 1 primary variables. TODO: split dimensions with only coefficients.
                continue;
            }
            auto primary = size.getPrimary();
            auto coefficient = size.getCoefficient();
            for (std::size_t primaryIndex = 0; primaryIndex < primaryCount; ++primaryIndex) {
                int primaryDim = primary[primaryIndex];
                if (primaryDim >= 1) {
                    auto primaryRes = Size(primaryCount, coefficientCount);
                    // Splitting out power of one is enough. For more, use more MergeOp's.
                    primaryRes.getPrimary()[primaryIndex] = 1;
                    bool canBeDividedByPrimary = size.quotientIsLegal(primaryRes);
                    if (canBeDividedByPrimary) {
                        checkThenAdd(outputShape[i], primaryRes);
                    }
                    for (std::size_t coefficientIndex = 0; coefficientIndex < coefficientCount; ++coefficientIndex) {
                        std::size_t coefficientDim = coefficient[coefficientIndex];
                        if (coefficientDim != 0) {
                            auto coefRes = Size(primaryRes);
                            // Here we simply split out the coefficient in half. TODO: better sampling.
                            if (canBeDividedByPrimary) {
                                coefRes.getCoefficient()[coefficientIndex] = coefficientDim > 0 ? ((coefficientDim + 1) / 2) : ((coefficientDim - 1) / 2);
                            } else {
                                coefRes.getCoefficient()[coefficientIndex] = coefficientDim > 0 ? ((coefficientDim + 1) / 2) : coefficientDim;
                            }
                            if (size.quotientIsLegal(coefRes)) {
                                checkThenAdd(outputShape[i], coefRes);
                            }
                        }
                    }
                }
            }
        }
    }
    return res;
}

} // namespace kas
