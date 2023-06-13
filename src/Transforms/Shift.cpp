#include "KAS/Transforms/Shift.hpp"


namespace kas {

std::size_t ShiftOp::initialHash() const noexcept {
    std::size_t h = std::hash<DimensionType>{}(Type);
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

} // namespace kas
