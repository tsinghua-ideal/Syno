#include "KAS/Transforms/Shift.hpp"


namespace kas {

std::size_t ShiftOp::initialHash() const noexcept {
    std::size_t h = static_cast<std::size_t>(Type);
    HashCombine(h, shift);
    return h;
}

IteratorValue ShiftOp::value(const IteratorValue& output) const {
    auto imm = ImmediateValueNode::Create(shift);
    auto size = ConstValueNode::Create(this->output.size());
    return (output + imm) % size;
}

bool ShiftOp::transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const {
    auto& out = interface[output];
    colors.substitute(interface, output, { getInput(), out.color });
    colors.simplify(interface); // Actually not needed.
    return true;
}

} // namespace kas
