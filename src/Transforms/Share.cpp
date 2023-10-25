#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

Color ShareOp::Input::computeColor(const GraphBuilder& graphBuilder) const {
    // Add constraint.
    return MergeLikeOp::Input::computeColor(graphBuilder).addTag(op);
}

ShareOp::ShareOp(const Dimension& output):
    MergeLikeOp { output },
    inputLhs { this, Order::Left },
    inputRhs { this, Order::Right }
{}

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

std::vector<const ShareOp *> ShareOp::Generate(PrimitiveOpStore& store, const Topmost& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    const Graph& graph = options.graph;

    // "Chained" Share.
    using enum DimensionTypeWithOrder;
    auto plausible = interface.filterOut({ ShareR, Split, Shift });

    std::vector<const ShareOp *> result;
    CountGenerateAttempts += interface.getDimensions().size();
    std::size_t countPlausible = 0;
    for (auto&& dim: plausible) {
        const auto& color = graph.colorOf(dim);
        // Since RHS will be a weight dim, we cannot make it data-discarding.
        if (color.isDataDiscarding()) continue;
        ++countPlausible;
        if (!options.allowance.shareWithinAllowance(dim.size())) {
            ++CountAllowanceExceeded;
            continue;
        }
        if (dim.is(DimensionType::Share)) {
            // We can assert that this is left, because we have filtered ShareR out!
            auto& self = dim.as<ShareOp::Input>();
            KAS_ASSERT(self.getOrder() == Order::Left);
        }
        if (color.size() + 1 > options.maxColorTags()) {
            // Too many color tags.
            ++CountMaximumTensorsExceeded;
            continue;
        }
        ++CountSuccessfulGenerations;
        result.emplace_back(store.get<ShareOp>(dim));
    }
    CountDisallowedAttempts += interface.getDimensions().size() - countPlausible;
    return result;
}

} // namespace kas
