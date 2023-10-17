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

std::unique_ptr<TensorView> FinalizeOp::buildTensorView(const std::vector<FixedDimension>& fixed, TensorExpression blending, const BindingContext& ctx) const {
    if (fixed.empty()) {
        return std::make_unique<TensorView>(tensors, std::move(blending), ctx);
    }
    std::vector<Topmost> tensors;
    std::ranges::copy(this->tensors, std::back_inserter(tensors));
    auto& inputTensor = tensors.at(0).getDimensions();
    for (const auto& [index, dim]: fixed) {
        // Given the fact that fixed is sorted.
        inputTensor.insert(inputTensor.begin() + index, dim);
    }
    return std::make_unique<TensorView>(tensors, std::move(blending), ctx);
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

bool FinalizeOp::FitIntoWeights(const std::vector<Dimension>& current, const WeightOptions& options) {
    if (current.empty()) {
        return true;
    }
    if (options.maximumTensors == 1) {
        return current.empty();
    }
    // The number of weights can be as small as greates the number of tags in ShareR's.
    return std::ranges::max(
        current
        | std::views::transform([](const Dimension& dim) {
            return dim.getColor().size();
        })
    ) + 1 <= options.maximumTensors;
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

std::size_t FinalizeOp::Distance(const std::vector<Dimension>& current, const Shape& desired, const ShapeComplexity::DistanceOptions& options) {
    int strideDist = 0;

    std::vector<Size> mustBeInput, canBeWeight;
    for (const Dimension& dim: current) {
        auto origin = dim.deduceOrigin();
        switch (origin) {
        case Dimension::Origin::Input:
            mustBeInput.emplace_back(dim.size());
            break;
        case Dimension::Origin::InputOrWeight:
            canBeWeight.emplace_back(dim.size());
            break;
        case Dimension::Origin::UnfoldOrExpand:
            ++strideDist; // We need an Unfold to eliminate this.
            break;
        default:
            KAS_CRITICAL("Dimension origin {} not allowed in FinalizeOp::Distance()!", origin);
        }
    }
    if (strideDist > options.remainingUnfoldsAndExpands) {
        return ShapeComplexity::Infinity; // Early stop.
    }

    // Then, experimentally finalize.
    std::size_t minimumComplexity = ShapeComplexity::Infinity;
    std::vector<Size> newCurrent = mustBeInput;
    auto recursion = [&](const auto& self, std::size_t trialIndex) -> void {
        auto trial = ShapeComplexity::Compute(desired, newCurrent, {
            .ctx = options.ctx,
            .remainingMerges = options.remainingMerges,
            .remainingSplits = options.remainingSplits,
            .remainingUnfoldsAndExpands = options.remainingUnfoldsAndExpands - strideDist,
            .overflow = std::min(minimumComplexity, options.overflow), // In either cases, we do not need to further compute.
        });
        minimumComplexity = std::min(minimumComplexity, trial);
        if (trialIndex < canBeWeight.size()) {
            self(self, trialIndex + 1);
            newCurrent.emplace_back(canBeWeight[trialIndex]);
            self(self, trialIndex + 1);
            newCurrent.pop_back();
        }
    };
    recursion(recursion, 0);
    if (minimumComplexity == ShapeComplexity::Infinity) {
        return ShapeComplexity::Infinity;
    } else {
        return minimumComplexity + strideDist;
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
Generator<std::vector<std::vector<Dimension>>> FinalizeOp::AssignToWeights(const std::vector<ColoredDimension>& remaining, std::size_t maxWeights) {
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
                for (auto subproblem: AssignToWeights(newInterface, maxWeights - 1)) {
                    KAS_ASSERT(std::ranges::all_of(subproblem, [](const std::vector<Dimension>& weight) { return !weight.empty(); }));
                    KAS_ASSERT(std::transform_reduce(subproblem.begin(), subproblem.end(), static_cast<std::size_t>(0), std::plus<>(), [](const std::vector<Dimension>& weight) { return weight.size(); }) == newInterface.size());
                    KAS_ASSERT(!weight.empty());
                    subproblem.emplace_back(weight);
                    KAS_ASSERT(std::transform_reduce(subproblem.begin(), subproblem.end(), static_cast<std::size_t>(0), std::plus<>(), [](const std::vector<Dimension>& weight) { return weight.size(); }) == remaining.size());
                    co_yield std::move(subproblem);
                }
            }
            stack.pop();
            continue;
        }
        if (trial == State::WillTryWithThis) {
            if (fragment.canAccept(startIndex)) {
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
    std::pair<std::vector<Dimension>, std::vector<ColoredDimension>> toTensorAndWeightDims(const std::vector<Dimension>& interface) const {
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
                weight.emplace_back(interface[i]);
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
        std::unique_ptr<TensorView> tensorView;
        std::size_t flops = 0;
        void build(const BindingContext& ctx, const FinalizeOp::TensorViewBuilder& tensorViewBuilder) {
            KAS_ASSERT(!tensorView);
            tensorView = tensorViewBuilder(tensors);
            flops = tensorView->getFLOPs(ctx);
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
    const FinalizeOp::TensorViewBuilder& tensorViewBuilder;
    const std::size_t maxFLOPs;
    // Sorted by variance, from lowest to highest.
    std::vector<Finalization> topK;

    TopKFinalizations(const BindingContext& ctx, const Graph::DimensionSet& sharedWeightDims, std::size_t k, const FinalizeOp::TensorViewBuilder& tensorViewBuilder, std::size_t maxFLOPs):
        ctx { ctx }, sharedWeightDims { sharedWeightDims }, k { k }, tensorViewBuilder { tensorViewBuilder }, maxFLOPs { maxFLOPs } {}
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
        f.build(ctx, tensorViewBuilder);
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
    std::vector<std::pair<FinalizeOp, std::unique_ptr<TensorView>>> toResult() {
        std::vector<std::pair<FinalizeOp, std::unique_ptr<TensorView>>> result;
        result.reserve(topK.size());
        for (auto& f: topK) {
            result.emplace_back(std::move(f.tensors), std::move(f.tensorView));
        }
        return result;
    }
};

} // namespace

std::vector<std::pair<FinalizeOp, std::unique_ptr<TensorView>>> FinalizeOp::Generate(const GraphHandle& dimensionsAndExpansions, const Graph& graph, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    const std::vector<Dimension>& interface = dimensionsAndExpansions.getDimensions();
    const std::vector<const Expand *>& expansions = dimensionsAndExpansions.getExpansions();

    // First we perform a basic check. If any Dimension is data-discarding, then it is not a legal kernel.
    if (std::ranges::any_of(interface, [](const Dimension& dim) { return dim.getColor().isDataDiscarding(); })) {
        ++CountFailedInvocations;
        return {};
    }

    const Graph::DimensionSet sharedWeightDims = ExpandOp::GetSharedWeightDims(graph);

    TopKFinalizations result { options.ctx, sharedWeightDims, options.maximumFinalizations, options.tensorViewBuilder, options.maxFLOPs };
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
        const auto [inputTensor, weightDims] = inputCandidate.toTensorAndWeightDims(interface);
        KAS_ASSERT(inputTensor.size() == desired.size());
        KAS_ASSERT(weightDims.size() == interface.size() - desired.size());
        for (auto tensors: AssignToWeights(weightDims, options.maximumTensors - 1)) {
            // Check whether the results are a partition of interface.
            {
                KAS_ASSERT(std::transform_reduce(tensors.begin(), tensors.end(), static_cast<std::size_t>(0), std::plus<> {}, [](const auto& t) { return t.size(); }) == weightDims.size());
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
                if (it != tensors.end()) {
                    continue;
                }
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
                auto origin = cDim.deduceOrigin();
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
            auto origin = dim.deduceOrigin();
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
