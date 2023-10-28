#include <list>
#include <memory>
#include <numeric>
#include <stack>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Algorithm.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

const ShapeDistance ShapeDistance::Infinity = { ShapeComplexity::Infinity, MaxFLOPs };

std::string ShapeDistance::toString() const {
    if (steps == ShapeComplexity::Infinity) {
        return "ShapeDistance(Infinity)";
    } else {
        return fmt::format("ShapeDistance(steps={}, flops={})", steps, flops);
    }
}

namespace {

struct CollectedDecreaseAndShare {
    std::set<const ShareOp *> shares;
    std::set<const Reduce *> decreases;
};

struct DecreaseAndShareCollector: public BottomTopDimVisitor<DecreaseAndShareCollector, CollectedDecreaseAndShare> {
    using Collected = CollectedDecreaseAndShare;
    auto transform(const Iterator&) const -> Collected { return {}; }
    auto transform(const Reduce& reduce) const -> Collected { return { {}, { &reduce } }; }
    auto transform(const RepeatLikeOp::Input& dim) const -> Collected {
        return at(dim.getOp()->output);
    }
    auto transform(const SplitLikeOp::Input& dim) const -> Collected {
        auto left = at(dim.getOp()->outputLhs), right = at(dim.getOp()->outputRhs);
        left.shares.merge(std::move(right.shares));
        left.decreases.merge(std::move(right.decreases));
        return left;
    }
    auto transform(const MergeLikeOp::Input& dim) const -> std::pair<Collected, Collected> {
        auto result = at(dim.getOp()->output);
        if (auto share = dynamic_cast<const ShareOp::Input *>(&dim); share) {
            result.shares.emplace(share->getDerivedOp<ShareOp>());
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

TensorView FinalizeOp::buildTensorView(const std::vector<FixedDimension>& fixed, TensorExpression blending, const BindingContext& ctx) const {
    if (fixed.empty()) {
        return TensorView { tensors, std::move(blending), ctx };
    }
    std::vector<Topmost> tensors;
    std::ranges::copy(this->tensors, std::back_inserter(tensors));
    auto& inputTensor = tensors.at(0).getDimensions();
    for (const auto& [index, dim]: fixed) {
        // Given the fact that fixed is sorted.
        inputTensor.insert(inputTensor.begin() + index, dim);
    }
    return TensorView { tensors, std::move(blending), ctx };
}

bool FinalizeOp::operator==(const FinalizeOp& rhs) const noexcept {
    return tensors == rhs.tensors;
}

std::size_t FinalizeOp::hash() const noexcept {
    return NextFinalizeSlot::GetKey(tensors);
}

GraphHandle FinalizeOp::toGraphHandle() const {
    return GraphHandle::FromInterfaces(tensors);
}

double FinalizeOp::weightVariance(const ConcreteConsts& consts) const {
    if (tensors.size() <= 1) {
        return 0.0;
    }
    const double count = tensors.size() - 1;
    double elements = 1.0;
    double sum = 0.0;
    std::vector<double> weights;
    weights.reserve(tensors.size() - 1);
    for (std::size_t tId = 1; tId < tensors.size(); ++tId) {
        const auto& tensor = tensors[tId];
        ShapeView shape = tensor.getShape();
        double weight = shape.totalSize().eval<double>(consts);
        elements *= weight;
        sum += weight;
        weights.emplace_back(weight);
    }
    const double avg = sum / count;
    // The `count` is by def of variance. The second factor is due to different consts.
    const double normalization = count * std::pow(elements, 2.0 / count);
    return std::transform_reduce(weights.begin(), weights.end(), 0.0, std::plus<>{}, [&avg](double weight) {
        return (weight - avg) * (weight - avg);
    }) / normalization;
}

double FinalizeOp::weightVariance(const BindingContext& ctx) const {
    const auto& allConsts = ctx.getAllConsts();
    return std::transform_reduce(allConsts.begin(), allConsts.end(), 0.0, std::plus<>{}, [this](const auto& consts) {
        return weightVariance(consts);
    });
}

std::string FinalizeOp::description(const BindingContext& ctx) const {
    return Topmost::Description(tensors, ctx);
}

bool FinalizeOp::hasRedundantWeights(const Graph::DimensionSet& sharedWeightDims) const {
    return std::ranges::any_of(
        tensors | std::views::drop(1), // only weights.
        [&](const Topmost& tensor) {
            return std::ranges::all_of(
                tensor.getDimensions(),
                [&](const Dimension& dim) {
                    // All dims of this weight are shared among weights.
                    return sharedWeightDims.contains(dim);
                }
            );
        }
    );
}

std::vector<ColoredDimension> FinalizeOp::FLOPsGameOptions::buildFullWeightDims(const std::vector<ColoredDimension>& selectedWeightDims) const {
    auto it1 = weightDims.begin();
    auto it2 = selectedWeightDims.begin();
    std::vector<ColoredDimension> result;
    while (it1 != weightDims.end() && it2 != selectedWeightDims.end()) {
        if (it1->hash() < it2->hash()) {
            result.emplace_back(*it1);
            ++it1;
        } else {
            result.emplace_back(*it2);
            ++it2;
        }
    }
    while (it1 != weightDims.end()) {
        result.emplace_back(*it1);
        ++it1;
    }
    while (it2 != selectedWeightDims.end()) {
        result.emplace_back(*it2);
        ++it2;
    }
    return result;
}

ShapeDistance FinalizeOp::Distance(const std::vector<std::pair<Dimension, int>>& current, const Shape& desired, const Graph& graph, const ShapeComplexity::DistanceOptions& options, const FLOPsGameOptions& flopsOptions) {
    int strideDist = 0;

    std::vector<std::pair<Size, int>> mustBeInput, canBeWeight;
    std::vector<ColoredDimension> canBeWeightDims;
    for (const auto& [dim, remainingLength]: current) {
        auto origin = dim.deduceOrigin(graph);
        switch (origin) {
        case Dimension::Origin::Input:
            mustBeInput.emplace_back(dim.size(), remainingLength);
            break;
        case Dimension::Origin::InputOrWeight:
            canBeWeight.emplace_back(dim.size(), remainingLength);
            canBeWeightDims.emplace_back(graph, dim);
            break;
        case Dimension::Origin::UnfoldOrExpand:
            ++strideDist; // We need an Unfold to eliminate this.
            if (remainingLength <= 0) {
                return ShapeDistance::Infinity;
            }
            break;
        default:
            KAS_CRITICAL("Dimension origin {} not allowed in FinalizeOp::Distance()!", origin);
        }
    }
    if (strideDist > options.remainingUnfoldsAndExpands) {
        return ShapeDistance::Infinity; // Early stop.
    }

    // Then, experimentally finalize.
    
    ShapeDistance minimumComplexity = ShapeDistance::Infinity;
    std::vector<std::pair<Size, int>> newCurrent = mustBeInput;
    std::vector<ColoredDimension> selectedWeightDims;
    auto extendedGame = ExtendedFLOPsGame(options.ctx, flopsOptions.totalInputSize, graph);
    const bool checkFLOPs = flopsOptions.prune;
    auto recursion = [&](const auto& self, std::size_t trialIndex) -> void {
        // In either cases, we do not need to further compute.
        if (trialIndex == canBeWeight.size()) {
            const std::size_t overflow = std::min(minimumComplexity.steps, options.overflow);
            ShapeDistance trial = { ShapeComplexity::Compute(desired, newCurrent, {
                .ctx = options.ctx,
                .remainingMerges = options.remainingMerges,
                .remainingSplits = options.remainingSplits,
                .remainingUnfoldsAndExpands = options.remainingUnfoldsAndExpands - strideDist,
                .overflow = overflow,
            }), ShapeDistance::MaxFLOPs };
            if (trial.steps <= overflow) {
                std::size_t minFLOPs = std::numeric_limits<std::size_t>::max();
                // Collect all weights.
                auto allWeightDims = flopsOptions.buildFullWeightDims(selectedWeightDims);
                // Enumerate weight dim assignment.
                for (auto weights: AssignToWeights(allWeightDims, {
                    .maxWeights = MaxTensorsToMaxWeights(flopsOptions.maximumTensors),
                    .allowWeightPermutation = false,
                })) {
                    auto game = extendedGame.getGameWithWeights(weights);
                    auto gameFLOPs = game.FLOPs();
                    minFLOPs = std::min(minFLOPs, gameFLOPs);
                }
                if (checkFLOPs && minFLOPs > flopsOptions.maxFLOPs) {
                    // This is not a valid solution.
                    trial.steps = ShapeComplexity::Infinity;
                } else {
                    trial.flops = minFLOPs;
                }
            }
            minimumComplexity = std::min(minimumComplexity, trial);
        } else if (trialIndex < canBeWeight.size()) {
            self(self, trialIndex + 1);
            newCurrent.emplace_back(canBeWeight[trialIndex]);
            selectedWeightDims.emplace_back(canBeWeightDims[trialIndex]);
            self(self, trialIndex + 1);
            newCurrent.pop_back();
            selectedWeightDims.pop_back();
        } else {
            KAS_UNREACHABLE();
        }
    };
    recursion(recursion, 0);
    if (minimumComplexity.steps == ShapeComplexity::Infinity) {
        return ShapeDistance::Infinity;
    } else {
        minimumComplexity.steps += strideDist;
        return minimumComplexity;
    }
}

namespace {

struct WeightFragment {
    const std::vector<ColoredDimension>& interface;
    std::vector<bool> used;
    WeightColor current;

    WeightFragment(const std::vector<ColoredDimension>& interface):
        interface { interface },
        used(interface.size(), false)
    {
        KAS_ASSERT(std::ranges::all_of(interface, [](const ColoredDimension& dim) { return dim.color.countRightTags() <= 1; }));
    }

    bool canAccept(std::size_t i) const {
        return !used[i] && current.disjointWith(interface[i].color);
    }
    void accept(std::size_t i) {
        current.merge(interface[i].color);
        used[i] = true;
    }
    bool unused() const {
        return std::ranges::all_of(used, std::logical_not{});
    }

    std::pair<std::vector<Dimension>, std::vector<ColoredDimension>> split() const {
        std::pair<std::vector<Dimension>, std::vector<ColoredDimension>> result;
        auto& [weight, newInterface] = result;
        for (std::size_t i = 0; i < interface.size(); ++i) {
            if (used[i]) {
                weight.emplace_back(interface[i].dim);
            } else {
                newInterface.emplace_back(interface[i]);
                newInterface.back().removeAllRightTagsIn(current);
            }
        }
        return result;
    }

    static std::size_t MinimumWeights(const std::vector<ColoredDimension>& remaining) {
        if (remaining.empty()) {
            return 0;
        }
        std::size_t count = std::ranges::max(remaining | std::views::transform([](const ColoredDimension& cDim) { return cDim.color.countTags(); }));
        if (count == 0) {
            return 1;
        } else {
            return count;
        }
    }
};

} // namespace

// Require that the dimensions are sorted according to hash!
Generator<std::vector<std::vector<Dimension>>> FinalizeOp::AssignToWeightsImpl(const std::vector<ColoredDimension>& remaining, std::size_t maxWeights, std::optional<std::size_t> maxHashFirstDimension) {
    if (remaining.empty()) {
        co_yield {};
        co_return;
    }
    if (WeightFragment::MinimumWeights(remaining) > maxWeights) {
        co_return;
    }
    if (maxWeights == 0) {
        co_return;
    } else if (maxWeights == 1) {
        // Special handling for maxWeights == 1.
        // We can only assign all remaining dimensions to one weight.
        // Do not forget to check for maxHashFirstDimension.
        if (
            maxHashFirstDimension.has_value()
            && remaining[0].hash() > *maxHashFirstDimension
        ) {
            co_return;
        }
        std::vector<Dimension> weight;
        WeightColor weightColor;
        for (const auto& cDim: remaining) {
            if (!weightColor.disjointWith(cDim.color)) {
                co_return;
            }
            weightColor.merge(cDim.color);
            weight.emplace_back(cDim.dim);
        }
        std::vector<std::vector<Dimension>> weights { std::move(weight) };
        co_yield std::move(weights);
        co_return;
    }

    // Now we have remaining.size() >= 1 and maxWeights >= 2.
    // Recursively find all possible assignments.

    struct State {
        enum Trial: bool {
            WillTryWithThis,
            WillTryWithoutThis,
        };
        std::size_t startIndex;
        Trial trial;
        WeightFragment fragment;
    };
    std::stack<State> stack;
    stack.emplace(0, State::WillTryWithThis, remaining);
    while (!stack.empty()) {
        auto& [startIndex, trial, fragment] = stack.top();
        if (startIndex == remaining.size()) {
            const auto [weight, newInterface] = fragment.split();
            if (!weight.empty()) {
                for (auto subproblem: AssignToWeightsImpl(newInterface, maxWeights - 1, weight[0].hash())) {
                    KAS_ASSERT(std::ranges::all_of(subproblem, [](const std::vector<Dimension>& weight) { return !weight.empty(); }));
                    KAS_ASSERT(std::transform_reduce(subproblem.begin(), subproblem.end(), 0_uz, std::plus<>(), [](const std::vector<Dimension>& weight) { return weight.size(); }) == newInterface.size());
                    KAS_ASSERT(!weight.empty());
                    subproblem.emplace_back(weight);
                    KAS_ASSERT(std::transform_reduce(subproblem.begin(), subproblem.end(), 0_uz, std::plus<>(), [](const std::vector<Dimension>& weight) { return weight.size(); }) == remaining.size());
                    co_yield std::move(subproblem);
                }
            }
            stack.pop();
            continue;
        }
        if (trial == State::WillTryWithThis) {
            if (
                fragment.canAccept(startIndex)
                && (
                    !maxHashFirstDimension.has_value()
                    || !fragment.unused()
                    || remaining[startIndex].dim.hash() <= *maxHashFirstDimension
                )
            ) {
                auto newFragment = fragment;
                newFragment.accept(startIndex);
                trial = State::WillTryWithoutThis;
                stack.emplace(startIndex + 1, State::WillTryWithThis, std::move(newFragment));
            } else {
                trial = State::WillTryWithoutThis;
            }
        } else if (trial == State::WillTryWithoutThis) {
            startIndex += 1;
            trial = State::WillTryWithThis;
        }
    }
    co_return;
}

Generator<std::vector<std::vector<Dimension>>> FinalizeOp::AssignToWeights(const std::vector<ColoredDimension>& weightDims, WeightOptions options) {
    KAS_ASSERT(std::ranges::is_sorted(weightDims | std::views::transform(&ColoredDimension::hash), std::less{}));
    return AssignToWeightsImpl(
        weightDims,
        options.maxWeights,
        options.allowWeightPermutation ? std::nullopt : std::make_optional(std::numeric_limits<std::size_t>::max())
    );
}

namespace {

struct CollectedTensorFragments {
    std::vector<std::size_t> mappings;
    std::vector<bool> used;
    CollectedTensorFragments(std::size_t size): used(size, false) {}
    bool canAccept(std::size_t index) const {
        // Collect tags.
        return std::ranges::find(mappings, index) == mappings.end() && !used[index];
    }
    void accept(std::size_t index) {
        mappings.emplace_back(index);
        used[index] = true;
    }
    std::pair<std::vector<Dimension>, std::vector<ColoredDimension>> toTensorAndWeightDims(const Graph& graph, const std::vector<Dimension>& interface) const {
        std::pair<std::vector<Dimension>, std::vector<ColoredDimension>> result;
        auto& [tensor, weight] = result;
        tensor.reserve(mappings.size());
        for (auto mapping: mappings) {
            tensor.emplace_back(interface[mapping]);
            KAS_ASSERT(used[mapping]);
        }
        weight.reserve(interface.size() - mappings.size());
        for (std::size_t i = 0; i < interface.size(); ++i) {
            if (!used[i]) {
                weight.emplace_back(graph, interface[i]);
            }
        }
        KAS_ASSERT(weight.size() == interface.size() - mappings.size());
        return result;
    }
};

struct TopKFinalizations {
    struct Finalization {
        FinalizeOp tensors;
        double variance;
        // Later manually assign this.
        std::unique_ptr<FinalStage> stage;
        std::size_t flops = 0;
        void build(const BindingContext& ctx, const FinalizeOp::FinalStageBuilder& finalStageBuilder) {
            KAS_ASSERT(!stage);
            stage = finalStageBuilder(tensors);
            flops = stage->value.getFLOPs(ctx);
        }
        Finalization(const BindingContext& ctx, auto&& tensors):
            tensors { std::forward<decltype(tensors)>(tensors) },
            variance { this->tensors.weightVariance(ctx) }
        {}
        std::weak_ordering operator<=>(const Finalization& other) const {
            auto count = tensors.count() <=> other.tensors.count();
            if (count != 0) {
                return count;
            }
            std::weak_ordering var = std::weak_ordering::equivalent;
            if (variance < other.variance) {
                var = std::weak_ordering::less;
            } else if (variance > other.variance) {
                var = std::weak_ordering::greater;
            }
            if (var != 0) {
                return var;
            }
            return flops <=> other.flops;
        }
    };
    const BindingContext& ctx;
    const Graph::DimensionSet& sharedWeightDims;
    const std::size_t k;
    const FinalizeOp::FinalStageBuilder& finalStageBuilder;
    const std::size_t maxFLOPs;
    // Sorted by variance, from lowest to highest.
    std::vector<Finalization> topK;

    TopKFinalizations(const BindingContext& ctx, const Graph::DimensionSet& sharedWeightDims, std::size_t k, const FinalizeOp::FinalStageBuilder& finalStageBuilder, std::size_t maxFLOPs):
        ctx { ctx }, sharedWeightDims { sharedWeightDims }, k { k }, finalStageBuilder { finalStageBuilder }, maxFLOPs { maxFLOPs } {}
    bool empty() const noexcept { return topK.empty(); }
    std::size_t size() const noexcept { return topK.size(); }
    void emplace(auto&& tensors) {
        Finalization f { ctx, std::forward<decltype(tensors)>(tensors) };
        if (f.tensors.hasRedundantWeights(sharedWeightDims)) {
            return;
        }

        // If the top-k cannot accomodate this, no need to check whether the result is within FLOPs by building.
        auto vacancy = std::lower_bound(topK.cbegin(), topK.cend(), f);
        if (vacancy == topK.cend() && topK.size() >= k) {
            return;
        }
        f.build(ctx, finalStageBuilder);
        if (f.flops > maxFLOPs) {
            return;
        }

        // Start again, so we can compare FLOPs.
        auto it = std::lower_bound(topK.begin(), topK.end(), f);
        if (it != topK.end()) {
            topK.insert(it, std::move(f));
            if (topK.size() > k) {
                topK.pop_back();
            }
        } else if (topK.size() < k) {
            topK.emplace_back(std::move(f));
        }
    }
    std::vector<std::pair<FinalizeOp, std::unique_ptr<FinalStage>>> toResult() {
        std::vector<std::pair<FinalizeOp, std::unique_ptr<FinalStage>>> result;
        result.reserve(topK.size());
        for (auto& f: topK) {
            result.emplace_back(std::move(f.tensors), std::move(f.stage));
        }
        return result;
    }
};

} // namespace

std::vector<std::pair<FinalizeOp, std::unique_ptr<FinalStage>>> FinalizeOp::Generate(const GraphHandle& dimensionsAndExpansions, const Graph& graph, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    const std::vector<Dimension>& interface = dimensionsAndExpansions.getDimensions();
    const std::vector<const Expand *>& expansions = dimensionsAndExpansions.getExpansions();

    // First we perform a basic check. If any Dimension is data-discarding, then it is not a legal kernel.
    if (std::ranges::any_of(interface, [&graph](const Dimension& dim) { return graph.colorOf(dim).isDataDiscarding(); })) {
        ++CountFailedInvocations;
        return {};
    }

    const Graph::DimensionSet sharedWeightDims = ExpandOp::GetSharedWeightDims(graph);

    TopKFinalizations result { options.ctx, sharedWeightDims, options.maximumFinalizations, options.finalStageBuilder, options.maxFLOPs };
    auto addToResults = [&result, &expansions](std::vector<std::vector<Dimension>>&& tensors) {
        // Currently, we only allow expansions to be added to the input tensor.
        std::vector<Topmost> realTensors;
        realTensors.emplace_back(std::move(tensors.at(0)), expansions);
        for (auto&& tensor: tensors | std::views::drop(1)) {
            realTensors.emplace_back(std::move(tensor), std::vector<const Expand *>{});
        }
        result.emplace(std::move(realTensors));
    };
    const auto& desired = options.desired;

    auto buildBesideInputTensor = [&](const CollectedTensorFragments& inputCandidate) {
        const auto [inputTensor, weightDims] = inputCandidate.toTensorAndWeightDims(graph, interface);
        KAS_ASSERT(inputTensor.size() == desired.size());
        KAS_ASSERT(weightDims.size() == interface.size() - desired.size());
        for (auto tensors: AssignToWeights(weightDims, {
            .maxWeights = MaxTensorsToMaxWeights(options.maximumTensors), 
            .allowWeightPermutation = options.allowWeightPermutation,
        })) {
            // Check whether the results are a partition of interface.
            {
                KAS_ASSERT(std::transform_reduce(tensors.begin(), tensors.end(), 0_uz, std::plus<> {}, [](const auto& t) { return t.size(); }) == weightDims.size());
                auto solution = inputTensor;
                std::ranges::copy(tensors | std::views::join, std::back_inserter(solution));
                std::ranges::sort(solution, Dimension::HashLessThan{});
                KAS_ASSERT(interface.size() == solution.size());
                KAS_ASSERT(std::ranges::equal(interface, solution));
            }
            // Canonicalize order of tensors.
            if (!options.allowWeightPermutation) {
                auto it = std::ranges::adjacent_find(tensors, [](const auto& a, const auto& b) {
                    return a[0].hash() > b[0].hash();
                });
                KAS_ASSERT(it == tensors.end());
            }
            tensors.insert(tensors.begin(), std::move(inputTensor));
            addToResults(std::move(tensors));
        }
    };

    auto collectInputDimensions = [&](const auto& self, std::size_t nextIndex, const CollectedTensorFragments& fragments) -> void {
        if (nextIndex == desired.size()) {
            // Carry out a simple check of whether all the must-be-input-dims have been collected.
            for (std::size_t i = 0; i < interface.size(); ++i) {
                if (fragments.used[i]) continue;
                const auto& cDim = interface[i];
                auto origin = cDim.deduceOrigin(graph);
                if (origin != Dimension::Origin::Weight && origin != Dimension::Origin::InputOrWeight) {
                    ++CountUncanonicalWeight;
                    return;
                }
            }
            // We have collected the full input shape. Now build the weights.
            buildBesideInputTensor(fragments);
            return;
        }
        const auto& desiredDimSize = desired[nextIndex];
        for (std::size_t i = 0; i < interface.size(); ++i) {
            auto&& dim = interface[i];
            auto origin = dim.deduceOrigin(graph);
            if (origin != Dimension::Origin::Input && origin != Dimension::Origin::InputOrWeight) {
                continue;
            }
            if (dim.size() == desiredDimSize && fragments.canAccept(i)) {
                auto newFragments = fragments;
                newFragments.accept(i);
                self(self, nextIndex + 1, newFragments);
            }
        }
    };
    collectInputDimensions(collectInputDimensions, 0, interface.size());

    CountLegalFinalizations += result.size();
    if (result.empty()) {
        ++CountFailedInvocations;
    } else {
        ++CountSuccessfulInvocations;
    }
    return result.toResult();
}

} // namespace kas
