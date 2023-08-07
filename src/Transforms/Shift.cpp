#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Shift.hpp"


namespace kas {

ShiftOp::ShiftOp(const Dimension& output, int shift):
    RepeatLikeOp { output },
    shift { shift },
    input { this }
{}

std::size_t ShiftOp::initialHash() const noexcept {
    std::size_t h = DimensionTypeHash(Type);
    HashCombine(h, shift);
    return h;
}

ShiftOp::Values ShiftOp::value(const Values& known) const {
    if (known.canSkipDeduction()) return known;
    auto& [input, output] = known.values;
    auto imm = ImmediateValueNode::Create(shift);
    auto size = ConstValueNode::Create(this->output.size());
    if (auto outputV = output.tryValue(); outputV) {
        // Out value -> in value.
        if (input.isUnorientedOrOrientedUp()) { // Check.
            return {{ (outputV + imm) % size, outputV }};
        }
    } else if (auto inputV = input.tryValue(); inputV) {
        // In value -> out value.
        if (output.isUnorientedOrOrientedDown()) { // Check.
            return {{ inputV, (inputV - imm) % size }};
        }
    } else if (output.isOrientedUp()) {
        // Out orientation -> in orientation.
        if (input.isUnorientedOrOrientedUp()) {
            return {{ Direction::Up, Direction::Up }};
        }
    } else if (input.isOrientedDown()) {
        // In orientation -> out orientation.
        if (output.isUnorientedOrOrientedDown()) {
            return {{ Direction::Down, Direction::Down }};
        }
    }
    // Otherwise, conflict.
    KAS_CRITICAL("Conflicting values for ShiftOp: input = {}, output = {}", input, output);
}

std::vector<const ShiftOp *> ShiftOp::Generate(PrimitiveOpStore& store, const GraphHandle& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    using enum DimensionTypeWithOrder;
    std::vector<DimensionTypeWithOrder> disallows { MapReduce, ShareL, ShareR, Shift };
    if (options.disallowShiftAboveUnfold) disallows.push_back(Unfold);
    auto plausible = interface.filterOut(disallows);

    std::vector<const ShiftOp *> result;
    CountGenerateAttempts += interface.getDimensions().size();
    std::size_t countPlausible = 0;
    for (auto&& dim: plausible) {
        ++countPlausible;
        ++CountSuccessfulGenerations;
        result.emplace_back(store.get<ShiftOp>(dim, 1));
    }
    CountDisallowedAttempts += interface.getDimensions().size() - countPlausible;
    return result;
}

} // namespace kas
