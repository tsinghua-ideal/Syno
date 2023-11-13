#include <list>
#include <memory>
#include <numeric>
#include <stack>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Transforms/Canonicalization.hpp"
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

std::vector<ColoredDimension> FinalizeOp::FLOPsGameOptions::buildFullWeightDims(const std::vector<ColoredDimension>& canBeWeightDims, const std::vector<bool>& select) const {
    KAS_ASSERT(canBeWeightDims.size() == select.size());
    auto it1 = weightDims.begin();
    auto it2 = 0_uz;
    std::vector<ColoredDimension> result;
    while (it1 != weightDims.end() && it2 != canBeWeightDims.size()) {
        if (!select[it2]) {
            ++it2;
            continue;
        }
        if (it1->hash() < canBeWeightDims[it2].hash()) {
            result.emplace_back(*it1);
            ++it1;
        } else {
            result.emplace_back(canBeWeightDims[it2]);
            ++it2;
        }
    }
    while (it1 != weightDims.end()) {
        result.emplace_back(*it1);
        ++it1;
    }
    while (it2 != canBeWeightDims.size()) {
        if (!select[it2]) {
            ++it2;
            continue;
        }
        result.emplace_back(canBeWeightDims[it2]);
        ++it2;
    }
    return result;
}

ShapeDistance FinalizeOp::Distance(const std::vector<CurrentDimension>& current, const std::vector<DesiredSize>& desired, const Graph& graph, const ShapeComplexity::DistanceOptions& options, const FLOPsGameOptions& flopsOptions) {
    ++CountShapeDistanceInvocations;

    int strideDist = 0;

    const BindingContext& ctx = options.ctx;

    std::vector<CurrentSize> mustBeInput;
    std::vector<Dimension> mustBeInputDims;
    for (const auto& dim: current) {
        auto origin = dim.value.deduceOrigin(graph);
        switch (origin) {
        case Dimension::Origin::Input:
        case Dimension::Origin::InputOrWeight: // We use Expand + Share to connect to Iterator in ContractionOp. So this is not needed.
            mustBeInput.emplace_back(dim);
            mustBeInputDims.emplace_back(dim);
            break;
        case Dimension::Origin::UnfoldOrExpand:
            ++strideDist; // We need an Unfold or an Expand to eliminate this.
            if (dim.remainingLength <= 0) {
                return ShapeDistance::Infinity;
            }
            break;
        default:
            KAS_CRITICAL("Dimension origin {} not allowed in FinalizeOp::Distance()!", origin);
        }
    }
    if (
        // Note that we can use Split to combine two strided dims, so the least requirement is 1.
        (strideDist > 0 && options.remainingUnfoldsAndExpands <= 0)
        // One step for each strided dim.
        || strideDist > options.overflow
    ) {
        return ShapeDistance::Infinity; // Early stop.
    }
    const std::size_t overflow = options.overflow - strideDist;

    // Then, experimentally finalize.

    ++CountShapeDistanceTrials;

    ShapeDistance trial = ShapeDistance::Infinity;

    auto unorderedness = ShapeComplexity::UnorderednessDeduction(desired, mustBeInput);
    auto extendedGame = ExtendedFLOPsGame(ctx, flopsOptions.totalInputSize, graph);
    auto enumerator = ShapeComplexity::Enumerator(desired, mustBeInput, {
        .ctx = ctx,
        .requiresOnlyOddNumelIncrease = options.requiresOnlyOddNumelIncrease,
        .remainingMerges = options.remainingMerges,
        .remainingSplits = options.remainingSplits,
        .remainingUnfoldsAndExpands = options.remainingUnfoldsAndExpands - (strideDist > 0),
        .overflow = overflow,
    }, &unorderedness);
    enumerator.enumerate();
    trial.steps = enumerator.getBestSteps();
    if (trial.steps > overflow) {
        ++CountShapeDistanceTrialTooManySteps;
        return ShapeDistance::Infinity;
    }

    unorderedness.intersects(enumerator.getUnorderedness());

    auto weights = AssignToWeights(flopsOptions.weightDims);
    auto game = extendedGame.getGameWithWeights(weights);
    trial.flops = game.FLOPs();
    if (flopsOptions.prune && trial.flops > flopsOptions.maxFLOPs) {
        // This is not a valid solution.
        ++CountShapeDistanceTrialTooManyFLOPs;
        return ShapeDistance::Infinity;
    }

    // Canonicalize by unorderedness. Check the unordered dims.
    Graph::DimensionMap<std::size_t> unorderedDims;
    // Collect all the unordered dims.
    for (const auto& deduction: unorderedness.get()) {
        for (std::size_t index: deduction.unorderedCurrent) {
            unorderedDims.try_emplace(mustBeInputDims.at(index), deduction.indexDesired);
        }
    }
    if (unorderedDims.empty()) {
        ++CountUnorderednessDeductionFailure;
    } else {
        ++CountUnorderednessDeductionSuccess;
    }
    if (!IsCanonicalGivenUnorderedness(graph, unorderedDims)) {
        ++CountShapeDistanceUnorderedCanonicalized;
        return ShapeDistance::Infinity;
    }
    // Although we may only need 1 Expand or Unfold to eliminate all strided dims (with the help of other ops),
    // we have to spend at least 1 step per strided dim.
    trial.steps += strideDist;
    return trial;
}

// Require that the dimensions are sorted according to hash!
std::vector<std::vector<Dimension>> FinalizeOp::AssignToWeights(const std::vector<ColoredDimension>& weightDims) {
    KAS_ASSERT(std::ranges::is_sorted(weightDims | std::views::transform(&ColoredDimension::hash), std::less{}));
    std::map<int, std::vector<Dimension>> weights;
    for (const auto& [dim, weightId]: weightDims) {
        weights[weightId.value()].emplace_back(dim);
    }
    return ranges::to<std::vector<std::vector<Dimension>>>(weights | std::views::values);
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
                const Dimension& dim = interface[i];
                auto shareR = dim.tryAs<ShareOp::Input>();
                std::optional<int> weightId = shareR ? std::make_optional(shareR->getRhsOrigin()) : std::nullopt;
                weight.emplace_back(dim, std::move(weightId));
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
    const std::size_t k;
    const FinalizeOp::FinalStageBuilder& finalStageBuilder;
    const std::size_t maxFLOPs;
    const std::size_t minFLOPs;
    // Sorted by variance, from lowest to highest.
    std::vector<Finalization> topK;

    TopKFinalizations(const BindingContext& ctx, std::size_t k, const FinalizeOp::FinalStageBuilder& finalStageBuilder, std::size_t maxFLOPs, std::size_t minFLOPs):
        ctx { ctx }, k { k }, finalStageBuilder { finalStageBuilder }, maxFLOPs { maxFLOPs }, minFLOPs { minFLOPs } {}
    bool empty() const noexcept { return topK.empty(); }
    std::size_t size() const noexcept { return topK.size(); }
    void emplace(auto&& tensors) {
        Finalization f { ctx, std::forward<decltype(tensors)>(tensors) };

        // If the top-k cannot accomodate this, no need to check whether the result is within FLOPs by building.
        auto vacancy = std::lower_bound(topK.cbegin(), topK.cend(), f);
        if (vacancy == topK.cend() && topK.size() >= k) {
            return;
        }
        f.build(ctx, finalStageBuilder);
        if (f.flops > maxFLOPs || f.flops < minFLOPs) {
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

    const BindingContext& ctx = options.ctx;
    const std::vector<Dimension>& interface = dimensionsAndExpansions.getDimensions();
    const std::vector<const Expand *>& expansions = dimensionsAndExpansions.getExpansions();

    // First we perform a basic check. If any Dimension is data-discarding, then it is not a legal kernel.
    if (std::ranges::any_of(interface, [&graph](const Dimension& dim) { return graph.colorOf(dim).isDataDiscarding(); })) {
        ++CountFailedInvocations;
        return {};
    }

    TopKFinalizations result { ctx, options.maximumFinalizations, options.finalStageBuilder, options.maxFLOPs, options.minFLOPs };
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

        // Pruning based on unorderedness.
        {
            Graph::DimensionMap<std::size_t> unorderedDims;
            for (std::size_t i = 0; const auto& desiredDim: desired) {
                if (desiredDim.isUnordered) {
                    unorderedDims.try_emplace(inputTensor.at(i), i);
                }
                ++i;
            }
            if (!IsCanonicalGivenUnorderedness(graph, unorderedDims)) {
                ++CountUncanonicalUnorderedInput;
                return;
            } else {
                ++CountCanonicalUnorderedInput;
            }
        }

        // TODO: If options.allowWeightPermutation, permute the weights.
        auto tensors = AssignToWeights(weightDims);
        // Check whether the results are a partition of interface.
        {
            KAS_ASSERT(std::transform_reduce(tensors.begin(), tensors.end(), 0_uz, std::plus<> {}, [](const auto& t) { return t.size(); }) == weightDims.size());
            auto solution = inputTensor;
            std::ranges::copy(tensors | std::views::join, std::back_inserter(solution));
            std::ranges::sort(solution, Dimension::HashLessThan{});
            KAS_ASSERT(interface.size() == solution.size());
            KAS_ASSERT(std::ranges::equal(interface, solution));
        }
        // Canonicalization of order of tensors is now done in the search.
        tensors.insert(tensors.begin(), inputTensor);
        addToResults(std::move(tensors));
    };

    auto collectInputDimensions = [&](const auto& self, std::size_t nextIndex, const CollectedTensorFragments& fragments) -> void {
        if (nextIndex == desired.size()) {
            // Carry out a simple check of whether all the must-be-input-dims have been collected.
            for (std::size_t i = 0; i < interface.size(); ++i) {
                if (fragments.used[i]) continue;
                const auto& cDim = interface[i];
                auto origin = cDim.deduceOrigin(graph);
                if (origin != Dimension::Origin::Weight) {
                    ++CountUncanonicalWeight;
                    return;
                }
            }
            // We have collected the full input shape. Now build the weights.
            buildBesideInputTensor(fragments);
            return;
        }
        const Size& desiredDimSize = desired[nextIndex].value;
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
