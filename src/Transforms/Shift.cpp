#include "KAS/Transforms/Shift.hpp"


namespace kas {

std::size_t ShiftOp::initialHash() const noexcept {
    std::size_t h = static_cast<std::size_t>(Type);
    HashCombine(h, shift);
    return h;
}

ShiftOp::IteratorValues ShiftOp::value(const IteratorValues& known) const {
    auto& [input, output] = known;
    auto imm = ImmediateValueNode::Create(shift);
    auto size = ConstValueNode::Create(this->output.size());
    if (!input && output) {
        return {{ .input = (output + imm) % size }};
    } else if (input && !output) {
        return {{ .output = (input - imm) % size }};
    } else {
        return {};
    }
}

ShiftOp::OrderingValues ShiftOp::ordering(const IteratorValues& known) const {
    return { .input = -1, .output = -1 };
}

std::size_t ShiftOp::CountColorTrials = 0;
std::size_t ShiftOp::CountColorSuccesses = 0;
bool ShiftOp::transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const {
    ++CountColorTrials;
    auto& out = interface[output];
    colors.substitute(interface, output, { getInput(), out.color });
    colors.simplify(interface); // Actually not needed.
    ++CountColorSuccesses;
    return true;
}

} // namespace kas
