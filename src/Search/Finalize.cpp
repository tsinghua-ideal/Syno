#include <list>
#include <memory>
#include <numeric>
#include <stack>

#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Algorithm.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

std::shared_ptr<TensorView> FinalizeOp::buildTensorView(const std::vector<FixedDimension>& fixed, TensorExpression blending) const {
    if (fixed.empty()) {
        return std::make_unique<TensorView>(tensors, std::move(blending));
    }
    std::vector<std::vector<Dimension>> tensors;
    std::ranges::copy(this->tensors, std::back_inserter(tensors));
    auto& inputTensor = tensors.at(0);
    for (const auto& [index, dim]: fixed) {
        // Given the fact that fixed is sorted.
        inputTensor.insert(inputTensor.begin() + index, dim);
    }
    return std::make_unique<TensorView>(tensors, std::move(blending));
}

bool FinalizeOp::operator==(const FinalizeOp& rhs) const noexcept {
    return tensors == rhs.tensors;
}

std::size_t FinalizeOp::hash() const noexcept {
    return NextFinalizeSlot::GetKey(tensors);
}

Dimensions FinalizeOp::toDimensions() const {
    auto result = ranges::to<Dimensions>(tensors | std::views::join);
    std::ranges::sort(result, Dimension::HashLessThan{});
    return result;
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
        ShapeView shape { tensor };
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
    if (allConsts.empty()) {
        return weightVariance(ctx.getDefaultConsts());
    }
    return std::transform_reduce(allConsts.begin(), allConsts.end(), 0.0, std::plus<>{}, [this](const auto& consts) {
        return weightVariance(consts);
    });
}

std::string FinalizeOp::description(const BindingContext& ctx) const {
    return TensorArrayToString(tensors, ctx);
}

bool FinalizeOp::Prune(const std::vector<Graph::ConnectedComponent>& components, const std::vector<std::vector<Dimension>>& trial) {
    std::map<Dimension, std::size_t, Dimension::AddressLessThan> dim2tensorId;
    for (std::size_t tId = 0; tId < trial.size(); ++tId) {
        for (auto&& dim: trial[tId]) {
            if (tId >= 1) {
                // We also check for uncanonical cases here.
                // In a single tensor, there must not be both inputs of MergeOp, or SplitOp, or ShiftOp.
                auto dimType = dim.type();
                switch (dimType) {
                case DimensionType::Merge: {
                    const auto& merge = dim.as<MergeOp::Input>();
                    if (auto it = dim2tensorId.find(merge.getOther()); it != dim2tensorId.end()) {
                        if (it->second == tId) {
                            return true;
                        }
                    }
                    break;
                }
                case DimensionType::Split:
                case DimensionType::Unfold:
                    // Unfold is not semantically equivalent to Split, but if we substitute it to a Split, we are essentially adding more parameters, so there always exists a valuation of weight such that Split and Unfold is equivalent here. In other words, Split can cover the semantics of Unfold.
                case DimensionType::Shift:
                    return true;
                default:
                    break;
                }
            }
            dim2tensorId[dim] = tId;
        }
    }

    // Early reduction analysis. For identity-mapped, sum-reduced, if a weight needs early reduction, then it is not canonical, which means we need to prune. TODO: if more types are added, change this.
    // We need to first identify the connected components in the indirected graph. If in a connected component, all output iterators are Sum, and all input iterators come from exactly one tensor, then this means we can do early reduction. For weight tensors, it is not reasonable to have early reduction, because this is pointless.
    for (auto&& component: components) {
        bool allSum = std::ranges::all_of(component.outputs, [](const Dimension& dim) {
            return dim.is(DimensionType::MapReduce);
        });
        if (!allSum) continue;
        const std::size_t tId = dim2tensorId.at(component.inputs.at(0));
        bool sameTensor = std::ranges::all_of(component.inputs | std::views::drop(1), [&](const Dimension& dim) {
            return tId == dim2tensorId.at(dim);
        });
        if (sameTensor) {
            // If this is from input tensor, then we can do early reduction to reduce FLOPs. TODO
            // But if this is from weight tensor, prune.
            if (tId != 0) {
                return true;
            }
        }
    }
    return false;
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

FinalizeOp::ReshapeGroup::ReshapeGroup(const Size& provision, const Size& consumption):
    remainder { provision / consumption },
    hasNoInput { false },
    splits { 0 },
    merges { 0 }
{}

FinalizeOp::ReshapeGroup::ReshapeGroup(const Size& provision):
    remainder { provision },
    hasNoInput { true },
    splits { 0 },
    merges { 0 }
{}

const Size& FinalizeOp::ReshapeGroup::getRemainder() const {
    return remainder;
}

void FinalizeOp::ReshapeGroup::addConsumption(const Size& consumption) {
    if (hasNoInput) {
        hasNoInput = false;
    } else {
        ++merges;
    }
    remainder = remainder / consumption;
}

void FinalizeOp::ReshapeGroup::addProvision(const Size& provision) {
    ++splits;
    remainder = remainder * provision;
}

bool FinalizeOp::ReshapeGroup::isLegal() const {
    auto trait = remainder.getTrait();
    return trait.value() != Size::Trait::IllegalCoefficient;
}

int FinalizeOp::ReshapeGroup::countSplits() const {
    return splits;
}

int FinalizeOp::ReshapeGroup::countTrivialMerges() const {
    return merges;
}

int FinalizeOp::ReshapeGroup::countFinalAdditionalMerges() const {
    if (hasNoInput) {
        return 0;
    }
    auto trait = remainder.getTrait();
    return static_cast<int>(trait.value() != Size::Trait::One);
}

int FinalizeOp::ReshapeGroup::countFinalUnfolds() const {
    if (hasNoInput) {
        return 1;
    }
    auto trait = remainder.getTrait();
    return static_cast<int>(trait.value() != Size::Trait::One);
}

FinalizeOp::ReshapeGroups::ReshapeGroups(const Shape& desired, const std::vector<Size>& current):
    desired { desired }, current { current },
    desiredToGroupId(desired.size(), NoGroup),
    currentToGroupId(current.size(), NoGroup),
    vacantCurrents { static_cast<int>(current.size()) }
{}

void FinalizeOp::ReshapeGroups::createGroup(std::size_t indexDesired, std::size_t indexCurrent) {
    KAS_ASSERT(!desiredAssigned(indexDesired) && !currentAssigned(indexCurrent));
    desiredToGroupId[indexDesired] = groups.size();
    currentToGroupId[indexCurrent] = groups.size();
    --vacantCurrents;
    groups.emplace_back(current[indexCurrent], desired[indexDesired]);
}

void FinalizeOp::ReshapeGroups::createGroup(std::size_t indexCurrent) {
    KAS_ASSERT(!currentAssigned(indexCurrent));
    currentToGroupId[indexCurrent] = groups.size();
    --vacantCurrents;
    groups.emplace_back(current[indexCurrent]);
}

void FinalizeOp::ReshapeGroups::addDesiredToGroup(std::size_t indexDesired, std::size_t indexGroup) {
    KAS_ASSERT(!desiredAssigned(indexDesired) && countGroups() > indexGroup);
    desiredToGroupId[indexDesired] = indexGroup;
    auto& group = groups[indexGroup];
    group.addConsumption(desired[indexDesired]);
}

void FinalizeOp::ReshapeGroups::addCurrentToGroup(std::size_t indexCurrent, std::size_t indexGroup) {
    KAS_ASSERT(!currentAssigned(indexCurrent) && countGroups() > indexGroup);
    currentToGroupId[indexCurrent] = indexGroup;
    --vacantCurrents;
    auto& group = groups[indexGroup];
    group.addProvision(current[indexCurrent]);
}

bool FinalizeOp::ReshapeGroups::desiredAssigned(std::size_t indexDesired) const {
    return desiredToGroupId[indexDesired] != NoGroup;
}
bool FinalizeOp::ReshapeGroups::currentAssigned(std::size_t indexCurrent) const {
    return currentToGroupId[indexCurrent] != NoGroup;
}
int FinalizeOp::ReshapeGroups::countVacantCurrents() const {
    return vacantCurrents;
}
int FinalizeOp::ReshapeGroups::countGroups() const {
    return groups.size();
}
int FinalizeOp::ReshapeGroups::countTrivialMerges() const {
    return FoldLeftFirst(groups | std::views::transform(&ReshapeGroup::countTrivialMerges), std::plus<>{}).value_or(0);
}
int FinalizeOp::ReshapeGroups::countSplits() const {
    return FoldLeftFirst(groups | std::views::transform(&ReshapeGroup::countSplits), std::plus<>{}).value_or(0);
}

auto FinalizeOp::ReshapeGroups::assignDesired(std::size_t indexDesired) const -> Generator<ReshapeGroups> {
    const Size& desiredSize = desired[indexDesired];
    KAS_ASSERT(desiredSize.getPrimaryPowersSum() == 1, "Input dimension sizes must be a single primary variable! TODO: support other shapes.");
    std::size_t varId = std::numeric_limits<std::size_t>::max();
    for (std::size_t pId = 0; auto p: desiredSize.getPrimary()) {
        if (p == 1) {
            varId = pId;
        }
        ++pId;
    }

    // First check for new sizes.
    for (std::size_t i = 0; i < current.size(); ++i) {
        if (currentAssigned(i)) continue;
        if (current[i].getPrimary()[varId] > 0) {
            auto copy = *this;
            copy.createGroup(indexDesired, i);
            co_yield std::move(copy);
        }
    }

    // Then check for provisions in existing groups.
    for (std::size_t i = 0; i < countGroups(); ++i) {
        if (groups[i].getRemainder().getPrimary()[varId] > 0) {
            auto copy = *this;
            copy.addDesiredToGroup(indexDesired, i);
            co_yield std::move(copy);
        }
    }
    co_return;
}

auto FinalizeOp::ReshapeGroups::assignCurrent(std::size_t indexCurrent) const -> Generator<ReshapeGroups> {
    // First add to existing groups.
    for (std::size_t gId = 0; gId < countGroups(); ++gId) {
        auto copy = *this;
        copy.addCurrentToGroup(indexCurrent, gId);
        co_yield std::move(copy);
    }

    // Then create new groups.
    auto copy = *this;
    copy.createGroup(indexCurrent);
    co_yield std::move(copy);
    co_return;
}

int FinalizeOp::ReshapeGroups::countIllegalGroups() const {
    return static_cast<int>(groups.size()) - static_cast<int>(std::ranges::count_if(groups, &ReshapeGroup::isLegal));
}

bool FinalizeOp::ReshapeGroups::isLegal() const {
    KAS_ASSERT(countVacantCurrents() == 0, "Must assign all sizes before calling isLegal()!");
    return std::ranges::all_of(groups, &ReshapeGroup::isLegal);
}

int FinalizeOp::ReshapeGroups::countFinalAdditionalMerges() const {
    KAS_ASSERT(countVacantCurrents() == 0, "Must assign all sizes before calling countFinalAdditionalMerges()!");
    return FoldLeftFirst(groups | std::views::transform(&ReshapeGroup::countFinalAdditionalMerges), std::plus<>{}).value_or(0);
}

int FinalizeOp::ReshapeGroups::countFinalUnfolds() const {
    KAS_ASSERT(countVacantCurrents() == 0, "Must assign all sizes before calling countFinalUnfolds()!");
    return FoldLeftFirst(groups | std::views::transform(&ReshapeGroup::countFinalUnfolds), std::plus<>{}).value_or(0);
}

std::size_t FinalizeOp::ReshapeGroups::countSteps() const {
    KAS_ASSERT(countVacantCurrents() == 0, "Must assign all sizes before calling countSteps()!");
    return countSplits() + countTrivialMerges() + countFinalAdditionalMerges() + countFinalUnfolds();
}

namespace {

template<typename T>
std::optional<T> optionalMin(const std::optional<T>& a, const std::optional<T>& b) {
    if (!a) return b;
    if (!b) return a;
    return std::min(*a, *b);
}

} // namespace

std::size_t FinalizeOp::ShapeComplexity(const Shape& desired, const std::vector<Size>& current, const FinalizeOp::DistanceOptions& options) {
    constexpr std::size_t Infinity = std::numeric_limits<std::size_t>::max();

    // First, check whether there are enough elements in the input tensor.
    if (current.empty()) {
        return Infinity;
    }
    Size desiredElements = desired.totalSize(), currentElements = Size::Product(current);
    auto totalQuotient = currentElements.canBeDividedBy(desiredElements);
    if (!totalQuotient || *totalQuotient == Size::Trait::IllegalCoefficient) {
        // Not enough elements.
        return Infinity;
    }

    // int numCurrent = static_cast<int>(current.size()), numDesired = static_cast<int>(desired.size());
    // Ideally, the groups are merge-split towers, i.e., reshapes.
    // Let there be N groups. Then the waist is N sizes.
    // N + #Splits = #current.
    // N + #Merges = #desired + #Unfolds.
    // N >= #Unfolds.
    // const int minGroups = std::max({
    //     0,
    //     numCurrent - options.remainingSplits,
    //     numDesired - options.remainingMerges,
    // });

    // Then group the dimensions recursively.
    ReshapeGroups root { desired, current };
    // Returns required steps or std::nullopt.
    auto currentMatchingRecursion = [&](const auto& self, const ReshapeGroups& groups, std::size_t currentIndex) -> std::optional<std::size_t> {
        if (
            groups.countTrivialMerges() > options.remainingMerges ||
            groups.countSplits() > options.remainingSplits
        ) {
            return std::nullopt;
        }
        // Need at least one current size to make a group legal.
        if (groups.countIllegalGroups() > groups.countVacantCurrents()) {
            return std::nullopt;
        }
        if (currentIndex != current.size()) {
            if (groups.currentAssigned(currentIndex)) {
                return self(self, groups, currentIndex + 1);
            }
            std::optional<std::size_t> result;
            for (ReshapeGroups newGroups: groups.assignCurrent(currentIndex)) {
                std::optional<std::size_t> newResult = self(self, newGroups, currentIndex + 1);
                result = optionalMin(result, newResult);
            }
            return result;
        } else { // We have grouped all sizes.
            if (!groups.isLegal()) {
                return std::nullopt;
            }
            if (groups.countFinalUnfolds() > options.remainingUnfolds || groups.countTrivialMerges() + groups.countFinalAdditionalMerges() > options.remainingMerges) {
                return std::nullopt;
            }
            return groups.countSteps();
        }
    };
    // Returns required steps or std::nullopt.
    auto desiredMatchingRecursion = [&](const auto& self, const ReshapeGroups& groups, std::size_t desiredIndex) -> std::optional<std::size_t> {
        if (
            groups.countTrivialMerges() > options.remainingMerges ||
            groups.countSplits() > options.remainingSplits
        ) {
            return std::nullopt;
        }
        if (desiredIndex != desired.size()) {
            std::optional<std::size_t> result;
            for (ReshapeGroups newGroups: groups.assignDesired(desiredIndex)) {
                std::optional<std::size_t> newResult = self(self, newGroups, desiredIndex + 1);
                result = optionalMin(result, newResult);
            }
            return result;
        } else { // We have assigned all desired sizes.
            return currentMatchingRecursion(currentMatchingRecursion, groups, 0);
        }
    };
    auto dist = desiredMatchingRecursion(desiredMatchingRecursion, root, 0);
    if (dist) {
        return *dist;
    } else {
        return Infinity;
    }
}

std::size_t FinalizeOp::Distance(const std::vector<Dimension>& current, const Shape& desired, const DistanceOptions& options) {
    constexpr std::size_t Infinity = std::numeric_limits<std::size_t>::max();

    int strideDist = 0;

    std::vector<Size> mustBeInput, canBeWeight;
    for (const Dimension& dim: current) {
        auto origin = dim.deduceOrigin();
        switch (origin) {
        case Dimension::Origin::Input:
            mustBeInput.emplace_back(dim.size());
            break;
        case Dimension::Origin::BothPossible:
            canBeWeight.emplace_back(dim.size());
            break;
        case Dimension::Origin::Unfold:
            ++strideDist; // We need an Unfold to eliminate this.
            break;
        default:
            KAS_CRITICAL("Dimension origin {} not allowed in FinalizeOp::Distance()!", origin);
        }
    }
    if (strideDist > options.remainingUnfolds) {
        return Infinity; // Early stop.
    }

    // Then, experimentally finalize.
    std::size_t minimumComplexity = Infinity;
    int newStrideDist = strideDist;
    std::vector<Size> newCurrent = mustBeInput;
    auto recursion = [&](const auto& self, std::size_t trialIndex) {
        if (newStrideDist > options.remainingUnfolds) {
            return;
        }
        auto trial = ShapeComplexity(desired, newCurrent, {
            .ctx = options.ctx,
            .remainingMerges = options.remainingMerges,
            .remainingSplits = options.remainingSplits,
            .remainingUnfolds = options.remainingUnfolds - newStrideDist,
        });
        minimumComplexity = std::min(minimumComplexity, trial);
        if (trialIndex < canBeWeight.size()) {
            self(self, trialIndex + 1);
            newStrideDist += 1;
            newCurrent.emplace_back(canBeWeight[trialIndex]);
            self(self, trialIndex + 1);
            newCurrent.pop_back();
            newStrideDist -= 1;
        }
    };
    recursion(recursion, 0);
    if (minimumComplexity == Infinity) {
        return Infinity;
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
            auto [weight, newInterface] = fragment.split();
            if (!weight.empty()) {
                for (auto subproblem: AssignToWeights(newInterface, maxWeights - 1)) {
                    subproblem.emplace_back(std::move(weight));
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
    std::pair<std::vector<Dimension>, std::vector<ColoredDimension>> toTensorAndWeightDims(const Dimensions& interface) const {
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
        Finalization(const BindingContext& ctx, auto&& tensors):
            tensors { std::forward<decltype(tensors)>(tensors) },
            variance { this->tensors.weightVariance(ctx) }
        {}
        std::weak_ordering operator<=>(const Finalization& other) const {
            auto count = tensors.count() <=> other.tensors.count();
            if (count != 0) {
                return count;
            }
            if (variance < other.variance) {
                return std::weak_ordering::less;
            } else if (variance > other.variance) {
                return std::weak_ordering::greater;
            } else {
                return std::weak_ordering::equivalent;
            }
        }
    };
    const BindingContext& ctx;
    std::size_t k;
    // Sorted by variance, from lowest to highest.
    std::vector<Finalization> topK;

    TopKFinalizations(const BindingContext& ctx, std::size_t k):
        ctx { ctx }, k { k } {}
    bool empty() const noexcept { return topK.empty(); }
    std::size_t size() const noexcept { return topK.size(); }
    void emplace(auto&& tensors) {
        Finalization f { ctx, std::forward<decltype(tensors)>(tensors) };
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
    std::vector<FinalizeOp> toResult() {
        std::vector<FinalizeOp> result;
        result.reserve(topK.size());
        for (auto& f: topK) {
            result.emplace_back(std::move(f.tensors));
        }
        return result;
    }
};

} // namespace

std::vector<FinalizeOp> FinalizeOp::Generate(const Dimensions& interface, const Graph& graph, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    // First we perform a basic check. If any Dimension is data-discarding, then it is not a legal kernel.
    if (std::ranges::any_of(interface, [](const Dimension& dim) { return dim.getColor().isDataDiscarding(); })) {
        ++CountFailedInvocations;
        return {};
    }

    // Compute connected components for early reduction analysis.
    auto components = graph.computeConnectedComponents();

    TopKFinalizations result { options.ctx, options.maximumFinalizations };
    const auto& desired = options.desired;

    auto buildBesideInputTensor = [&](const CollectedTensorFragments& inputCandidate) {
        const auto [inputTensor, weightDims] = inputCandidate.toTensorAndWeightDims(interface);
        KAS_ASSERT(inputTensor.size() == desired.size());
        KAS_ASSERT(weightDims.size() == interface.size() - desired.size());
        for (auto tensors: AssignToWeights(weightDims, options.maximumTensors - 1)) {
            // Check whether the results are a partition of interface.
            {
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
            if (!Color::CheckFinalization(tensors)) {
                // We really should avoid this!
                KAS_WARNING("Finalization with conflicting colors generated!");
                ++CountConflictingColors;
                return;
            }
            if (Prune(components, tensors)) {
                ++CountPrunedFinalizations;
                return;
            }
            result.emplace(std::move(tensors));
        }
    };

    auto collectInputDimensions = [&](const auto& self, std::size_t nextIndex, const CollectedTensorFragments& fragments) -> void {
        if (nextIndex == desired.size()) {
            // Carry out a simple check of whether all the must-be-input-dims have been collected.
            for (std::size_t i = 0; i < interface.size(); ++i) {
                if (fragments.used[i]) continue;
                const auto& cDim = interface[i];
                auto origin = cDim.deduceOrigin();
                if (origin != Dimension::Origin::Weight && origin != Dimension::Origin::BothPossible) {
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
            if (origin != Dimension::Origin::Input && origin != Dimension::Origin::BothPossible) {
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
