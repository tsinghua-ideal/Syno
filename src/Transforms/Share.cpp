#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/OperationStore.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

bool ShareOp::isEqual(const Operation& other) const {
    const ShareOp& rhs = static_cast<const ShareOp&>(other);
    return output == rhs.output && rhsOrigin == rhs.rhsOrigin;
}

Color ShareOp::Input::computeColor(const GraphBuilder& graphBuilder) const {
    // Add constraint.
    return MergeLikeOp::Input::computeColor(graphBuilder).addTag(op);
}

ShareOp::ShareOp(const Dimension& output, int rhsOrigin):
    MergeLikeOp { output },
    rhsOrigin { rhsOrigin },
    inputLhs { this, Order::Left },
    inputRhs { this, Order::Right }
{}

std::size_t ShareOp::initialHash() const noexcept {
    std::size_t h = DimensionTypeHash(Type);
    constexpr std::size_t
        maxTensors = 4,
        delta = std::numeric_limits<std::size_t>::digits / (2 * maxTensors),
        ones = (1_uz << delta) - 1;
    HashCombine(h, ones << (delta * static_cast<std::size_t>(rhsOrigin + maxTensors)));
    return h;
}

ShareOp::Values ShareOp::value(const Values& known) const {
    if (known.canSkipDeduction()) return known;
    auto& [inputLhs, inputRhs, output] = known.values;
    // In the following 3 cases, value propagates from one branch to the others.
    if (auto v = output.tryValue(); v) {
        if (inputLhs.isUnorientedOrOrientedUp() && inputRhs.isUnorientedOrOrientedUp()) { // Check.
            return {{ v, v, v }};
        }
    } else if (auto v = inputLhs.tryValue(); v) {
        if (inputRhs.isUnorientedOrOrientedUp() && output.isUnorientedOrOrientedDown()) { // Check.
            return {{ v, v, v }};
        }
    } else if (auto v = inputRhs.tryValue(); v) {
        if (inputLhs.isUnorientedOrOrientedUp() && output.isUnorientedOrOrientedDown()) { // Check.
            return {{ v, v, v }};
        }
    }
    // In the following 3 cases, orientation propagates from one branch to the others.
    else if (output.isOrientedUp()) {
        if (inputLhs.isUnorientedOrOrientedUp() && inputRhs.isUnorientedOrOrientedUp()) { // Check.
            return {{ Direction::Up, Direction::Up, Direction::Up }};
        }
    } else if (inputLhs.isOrientedDown()) {
        if (inputRhs.isUnorientedOrOrientedUp() && output.isUnorientedOrOrientedDown()) { // Check.
            return {{ Direction::Down, Direction::Up, Direction::Down }};
        }
    } else if (inputRhs.isOrientedDown()) {
        if (inputLhs.isUnorientedOrOrientedUp() && output.isUnorientedOrOrientedDown()) { // Check.
            return {{ Direction::Up, Direction::Down, Direction::Down }};
        }
    }
    // Otherwise, conficts.
    KAS_CRITICAL("Conflicting values for ShareOp: inputLhs = {}, inputRhs = {}, output = {}", inputLhs, inputRhs, output);
}

std::pair<bool, CompactColor> ShareOp::transformColor(CompactColor fro1, CompactColor fro2) const {
    // Require empty intersection.
    return { !(fro1 & fro2), fro1 | fro2 };
}

std::set<int> ShareOp::GetRhsOrigins(const Graph& graph) {
    std::set<int> result;
    for (const ShareOp *op: graph.getOpsOfType<ShareOp>()) {
        result.emplace(op->getRhsOrigin());
    }
    return result;
}

} // namespace kas
