#include "KAS/Transforms/Shift.hpp"


namespace kas {

std::size_t ShiftOp::Input::hash() const noexcept {
    auto h = static_cast<std::size_t>(type());
    HashCombine(h, op->output.hash());
    HashCombine(h, getDerivedOp<ShiftOp>()->shift);
    return h;
}

IteratorValue ShiftOp::value(const IteratorValue& output) const {
    auto imm = ImmediateValueNode::Create(shift);
    auto size = ConstValueNode::Create(this->output.size());
    return (output + imm) % size;
}

} // namespace kas
