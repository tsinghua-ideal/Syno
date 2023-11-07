#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

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

std::map<int, Dimension> ShareOp::GetWeightLeaders(const Graph& graph) {
    std::map<int, Dimension> weightIdToLeastDim;
    auto comp = Dimension::GlobalLessThan(graph);
    for (const ShareOp *op: graph.getOpsOfType<ShareOp>()) {
        auto [least, first] = weightIdToLeastDim.try_emplace(op->getRhsOrigin(), op->output);
        if (!first) {
            // Find the least dimension.
            if (comp(op->output, least->second)) {
                least->second = op->output;
            }
        }
    }
    return weightIdToLeastDim;
}

std::size_t ShareOp::LeastRemainingShares(const Topmost& interface, const Graph& graph) {
    const auto weightLeaders = GetWeightLeaders(graph);
    if (weightLeaders.empty()) return 0; // No weights.
    // The weights are 1...countWeights. It is possible that weightLeaders.size() < countWeights,
    // because there may be some weights not added yet.
    const int countWeights = weightLeaders.rbegin()->first;
    KAS_ASSERT(countWeights > 0 && weightLeaders.size() <= countWeights);
    KAS_ASSERT(weightLeaders.begin()->first > 0);

    auto leaders = ranges::to<std::vector<std::pair<int, Dimension>>>(weightLeaders);
    auto globalLessThan = Dimension::GlobalLessThan(graph);
    // Now sort the leaders. If they are in order (1, 2, 3, ...), then no other steps needed.
    std::ranges::sort(leaders, globalLessThan, &std::pair<int, Dimension>::second);

    int requiredAdditionalShares = 0;
    int currentlyOrderedWeights = 0;
    int currentlyOrderedToLeaderIndex = 0;
    while (currentlyOrderedWeights < countWeights) {
        const auto& leader = leaders[currentlyOrderedToLeaderIndex];
        int extra = leader.first - currentlyOrderedWeights - 1;
        KAS_ASSERT(extra >= 0);

        if (extra > 0) {
            requiredAdditionalShares += extra;
            // We also need to count whether there are enough slots for the extra Share's.
            const int leaderHeight = globalLessThan.heightOf(leader.second);
            int slots = 0;
            for (const Dimension& dim: interface.getDimensions()) {
                const int h = globalLessThan.heightOf(dim);
                if (h < leaderHeight) {
                    slots += leaderHeight - h;
                } else if (h == leaderHeight) {
                    slots += globalLessThan.hash(dim, leader.second);
                }
            }
            // Note that the slots contraint is compulsory, because the leader of this weight can only be decreased by filling into the slots.
            if (slots < requiredAdditionalShares) {
                return std::numeric_limits<std::size_t>::max();
            }
        }

        currentlyOrderedWeights = leader.first;

        while (
            currentlyOrderedToLeaderIndex < leaders.size() &&
            // If the leader of this weight has to be reordered to an earlier position, then it can be placed arbitrarily, so we can skip checking it.
            leaders[currentlyOrderedToLeaderIndex].first <= currentlyOrderedWeights
        ) {
            ++currentlyOrderedToLeaderIndex;
        }
    }

    // For example, if tensor 3 is the least, we need to move tensor 1 and tensor 2 before it.
    return requiredAdditionalShares;
}

std::vector<const ShareOp *> ShareOp::Generate(PrimitiveOpStore& store, const Topmost& interface, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    const Graph& graph = options.graph;

    // "Chained" Share.
    using enum DimensionTypeWithOrder;
    auto plausible = interface.filterOut({ ShareR });

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
        std::vector<bool> trials(options.maximumTensors, true);
        trials[0] = false;
        for (const MergeLikeOp *op: color.getTags()) {
            auto previous = dynamic_cast<const ShareOp *>(op);
            KAS_ASSERT(previous, "Tags in Color must be ShareOp.");
            trials.at(previous->getRhsOrigin()) = false;
        }
        for (std::size_t i = 1; i < options.maximumTensors; ++i) {
            if (trials[i]) {
                result.emplace_back(store.get<ShareOp>(dim, i));
            }
        }
    }
    CountDisallowedAttempts += interface.getDimensions().size() - countPlausible;
    return result;
}

} // namespace kas
