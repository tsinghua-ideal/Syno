#include "KAS/Search/FLOPsGame.hpp"
#include "KAS/Transforms/Transforms.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

namespace {

struct CollectedDecreaseAndShare {
    std::set<const ShareOp *> shares;
    std::set<const Reduce *> decreases;
};

struct DecreaseAndShareCollector: public BottomTopDimVisitor<DecreaseAndShareCollector, CollectedDecreaseAndShare> {
    using Collected = CollectedDecreaseAndShare;
    auto transform(const Iterator&) const -> Collected { return {}; }
    auto transform(const Reduce& reduce) const -> Collected { return { {}, { &reduce } }; }
    auto transform(const RepeatLikeOp& op) const -> Collected {
        return at(op.output);
    }
    auto transform(const SplitLikeOp& op) const -> Collected {
        auto left = at(op.outputLhs), right = at(op.outputRhs);
        left.shares.merge(std::move(right.shares));
        left.decreases.merge(std::move(right.decreases));
        return left;
    }
    auto transform(const MergeLikeOp& op) const -> std::pair<Collected, Collected> {
        auto result = at(op.output);
        if (auto shareOp = dynamic_cast<const ShareOp *>(&op); shareOp) {
            result.shares.emplace(shareOp);
        }
        return { result, result };
    }
};

struct NumelAdjacency {
    // Expand and Unfold.
    std::set<const PrimitiveOp *> increase;
    std::set<const Reduce *> decrease;
};

} // namespace

ExtendedFLOPsGame::ExtendedFLOPsGame(const BindingContext& ctx, Size inputSize, const Graph& graph):
    ctx { ctx },
    inputSize { std::move(inputSize) }
{
    // We assume that weights are only connected to Share RHS and Iterator.
    // `increase`s originates from Expand, Unfold and Iterators in weights.
    // `decrease`s originates from Reduce. TODO: consider Stride.
    DecreaseAndShareCollector collector;
    graph.accept(collector);
    // First find all the shares, and traverse the collected items to collect decreases.
    std::map<const ShareOp *, NumelAdjacency> sharedDependencies;
    for (const ShareOp *shareOp: graph.getOpsOfType<ShareOp>()) {
        const CollectedDecreaseAndShare& collected = collector.at(shareOp->output);
        sharedDependencies.try_emplace(shareOp, NumelAdjacency{{}, collected.decreases});
    }
    // Then find all the increases.
    std::map<const PrimitiveOp *, std::pair<Size, std::set<const Reduce *>>> increase;
    // Expand.
    for (const ExpandOp *expandOp: graph.getOpsOfType<ExpandOp>()) {
        const CollectedDecreaseAndShare& collected = collector.at(expandOp->output);
        for (const ShareOp *shareOp: collected.shares) {
            increase.try_emplace(expandOp, expandOp->output.size(), collected.decreases);
            sharedDependencies.at(shareOp).increase.emplace(expandOp);
        }
    }
    // Unfold.
    for (const UnfoldOp *unfoldOp: graph.getOpsOfType<UnfoldOp>()) {
        const CollectedDecreaseAndShare& collected = collector.at(unfoldOp->getInput());
        for (const ShareOp *shareOp: collected.shares) {
            increase.try_emplace(unfoldOp, unfoldOp->getWindow(), collected.decreases);
            sharedDependencies.at(shareOp).increase.emplace(unfoldOp);
        }
    }
    // Now we collect all the sizes.
    std::map<const PrimitiveOp *, std::size_t> increaseIndex;
    std::map<const Reduce *, std::size_t> decreaseIndex;
    for (const auto& [op, sizeAndDec]: increase) {
        const auto& [size, dec] = sizeAndDec;
        increaseIndex.emplace(op, increaseIndex.size());
        this->increase.emplace_back(size);
    }
    for (const Reduce *reduction: graph.getReduceIterators()) {
        decreaseIndex.emplace(reduction, decreaseIndex.size());
        this->decrease.emplace_back(reduction->getBase().getDomain());
    }
    // Then translate into indices.
    for (const auto& [shareOp, adj]: sharedDependencies) {
        const auto& [inc, dec] = adj;
        this->sharedDependencies.try_emplace(
            shareOp->getInputR(),
            ranges::to<std::vector<std::size_t>>(inc | std::views::transform([&](const PrimitiveOp *op) { return increaseIndex.at(op); })),
            ranges::to<std::vector<std::size_t>>(dec | std::views::transform([&](const Reduce *reduction) { return decreaseIndex.at(reduction); }))
        );
    }
    // Finally, the Iterators.
    for (const Dimension& input: graph.getTopmost().getDimensions()) {
        if (auto iterator = input.tryAs<Iterator>(); iterator) {
            this->sharedDependencies.try_emplace(input, ExtendedFLOPsGame::Adjacency {
                .increaseIndices = { this->increase.size() },
                .decreaseIndices = {},
            });
            this->increase.emplace_back(iterator->size());
        }
    }
    this->dependencies.resize(this->decrease.size(), std::vector<bool>(this->increase.size(), false));
    // Fill in the dependencies.
    for (const auto& [inc, sizeAndDec]: increase) {
        const auto& [size, dec] = sizeAndDec;
        for (const Reduce *reduction: dec) {
            dependencies[decreaseIndex.at(reduction)][increaseIndex.at(inc)] = true;
        }
    }
}

FLOPsGame ExtendedFLOPsGame::getGameWithWeights(const std::vector<std::vector<Dimension>>& weights) const {
    auto dependencies = this->dependencies;
    for (const std::vector<Dimension>& weight: weights) {
        std::set<std::size_t> requiredIncrease, involvedDecrease;
        for (const Dimension& weightDim: weight) {
            const Adjacency& adj = sharedDependencies.at(weightDim);
            std::ranges::copy(adj.increaseIndices, std::inserter(requiredIncrease, requiredIncrease.begin()));
            std::ranges::copy(adj.decreaseIndices, std::inserter(involvedDecrease, involvedDecrease.begin()));
        }
        for (std::size_t i: involvedDecrease) {
            for (std::size_t j: requiredIncrease) {
                dependencies[i][j] = true;
            }
        }
    }
    return FLOPsGame {
        .ctx = ctx,
        .inputSize = inputSize,
        .increase = increase,
        .decrease = decrease,
        .dependencies = std::move(dependencies),
    };
}

} // namespace kas
