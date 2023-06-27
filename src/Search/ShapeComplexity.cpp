#include "KAS/Core/Shape.hpp"
#include "KAS/Search/ShapeComplexity.hpp"


namespace kas::ShapeComplexity {

ReshapeGroup::ReshapeGroup(const Size& provision, const Size& consumption):
    remainder { provision / consumption },
    hasNoInput { false },
    splits { 0 },
    merges { 0 }
{}

ReshapeGroup::ReshapeGroup(const Size& provision):
    remainder { provision },
    hasNoInput { true },
    splits { 0 },
    merges { 0 }
{}

const Size& ReshapeGroup::getRemainder() const {
    return remainder;
}

void ReshapeGroup::addConsumption(const Size& consumption) {
    if (hasNoInput) {
        hasNoInput = false;
    } else {
        ++merges;
    }
    remainder = remainder / consumption;
}

void ReshapeGroup::addProvision(const Size& provision) {
    ++splits;
    remainder = remainder * provision;
}

bool ReshapeGroup::isLegal() const {
    auto trait = remainder.getTrait();
    return trait.value() != Size::Trait::IllegalCoefficient;
}

int ReshapeGroup::countSplits() const {
    return splits;
}

int ReshapeGroup::countTrivialMerges() const {
    return merges;
}

int ReshapeGroup::countFinalAdditionalMerges() const {
    if (hasNoInput) {
        return 0;
    }
    auto trait = remainder.getTrait();
    return static_cast<int>(trait.value() != Size::Trait::One);
}

int ReshapeGroup::countFinalUnfolds() const {
    if (hasNoInput) {
        return 1;
    }
    auto trait = remainder.getTrait();
    return static_cast<int>(trait.value() != Size::Trait::One);
}

ReshapeGroups::ReshapeGroups(const Shape& desired, const std::vector<Size>& current):
    desired { desired }, current { current },
    desiredToGroupId(desired.size(), NoGroup),
    currentToGroupId(current.size(), NoGroup),
    vacantCurrents { static_cast<int>(current.size()) }
{}

void ReshapeGroups::createGroup(std::size_t indexDesired, std::size_t indexCurrent) {
    KAS_ASSERT(!desiredAssigned(indexDesired) && !currentAssigned(indexCurrent));
    desiredToGroupId[indexDesired] = groups.size();
    currentToGroupId[indexCurrent] = groups.size();
    --vacantCurrents;
    groups.emplace_back(current[indexCurrent], desired[indexDesired]);
}

void ReshapeGroups::createGroup(std::size_t indexCurrent) {
    KAS_ASSERT(!currentAssigned(indexCurrent));
    currentToGroupId[indexCurrent] = groups.size();
    --vacantCurrents;
    groups.emplace_back(current[indexCurrent]);
}

void ReshapeGroups::addDesiredToGroup(std::size_t indexDesired, std::size_t indexGroup) {
    KAS_ASSERT(!desiredAssigned(indexDesired) && countGroups() > indexGroup);
    desiredToGroupId[indexDesired] = indexGroup;
    auto& group = groups[indexGroup];
    group.addConsumption(desired[indexDesired]);
}

void ReshapeGroups::addCurrentToGroup(std::size_t indexCurrent, std::size_t indexGroup) {
    KAS_ASSERT(!currentAssigned(indexCurrent) && countGroups() > indexGroup);
    currentToGroupId[indexCurrent] = indexGroup;
    --vacantCurrents;
    auto& group = groups[indexGroup];
    group.addProvision(current[indexCurrent]);
}

bool ReshapeGroups::desiredAssigned(std::size_t indexDesired) const {
    return desiredToGroupId[indexDesired] != NoGroup;
}
bool ReshapeGroups::currentAssigned(std::size_t indexCurrent) const {
    return currentToGroupId[indexCurrent] != NoGroup;
}
int ReshapeGroups::countVacantCurrents() const {
    return vacantCurrents;
}
int ReshapeGroups::countGroups() const {
    return groups.size();
}
ReshapeGroups::Counts ReshapeGroups::count() const {
    Counts counts {};
    for (const ReshapeGroup& group: groups) {
        counts.trivialMerges += group.countTrivialMerges();
        counts.splits += group.countSplits();
    }
    return counts;
}

auto ReshapeGroups::assignDesired(std::size_t indexDesired) const -> Generator<ReshapeGroups> {
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

auto ReshapeGroups::assignCurrent(std::size_t indexCurrent) const -> Generator<ReshapeGroups> {
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

ReshapeGroups::CurrentCounts ReshapeGroups::currentCount() const {
    CurrentCounts counts = { .illegalGroups = static_cast<int>(groups.size()) };
    for (const ReshapeGroup& group: groups) {
        counts.trivialMerges += group.countTrivialMerges();
        counts.splits += group.countSplits();
        counts.illegalGroups -= static_cast<int>(group.isLegal());
    }
    return counts;
}

bool ReshapeGroups::isLegal() const {
    KAS_ASSERT(countVacantCurrents() == 0, "Must assign all sizes before calling isLegal()!");
    return std::ranges::all_of(groups, &ReshapeGroup::isLegal);
}

ReshapeGroups::FinalCounts ReshapeGroups::finalCount() const {
    KAS_ASSERT(countVacantCurrents() == 0, "Must assign all sizes before calling finalCount()!");
    FinalCounts counts {};
    for (const ReshapeGroup& group: groups) {
        counts.trivialMerges += group.countTrivialMerges();
        counts.splits += group.countSplits();
        counts.additionalMerges += group.countFinalAdditionalMerges();
        counts.unfolds += group.countFinalUnfolds();
    }
    return counts;
}

int ReshapeGroups::FinalCounts::merges() const {
    return trivialMerges + additionalMerges;
}

std::size_t ReshapeGroups::FinalCounts::steps() const {
    return trivialMerges + splits + additionalMerges + unfolds;
}

namespace {

template<typename T>
std::optional<T> optionalMin(const std::optional<T>& a, const std::optional<T>& b) {
    if (!a) return b;
    if (!b) return a;
    return std::min(*a, *b);
}

} // namespace

std::size_t Compute(const Shape& desired, const std::vector<Size>& current, const DistanceOptions& options) {
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
        const auto& counts = groups.currentCount();
        const int &trivialMerges = counts.trivialMerges, &splits = counts.splits, &illegalGroups = counts.illegalGroups;
        if (
            trivialMerges > options.remainingMerges ||
            splits > options.remainingSplits
        ) {
            return std::nullopt;
        }
        // Need at least one current size to make a group legal.
        if (illegalGroups > groups.countVacantCurrents()) {
            return std::nullopt;
        }
        // Note that in this stage, each vacant current size accounts for at least one step.
        if (trivialMerges + splits + groups.countVacantCurrents() > options.overflow) {
            return Infinity;
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
            const auto finalCounts = groups.finalCount();
            if (finalCounts.unfolds > options.remainingUnfolds || finalCounts.merges() > options.remainingMerges) {
                return std::nullopt;
            }
            return finalCounts.steps();
        }
    };
    // Returns required steps or std::nullopt.
    auto desiredMatchingRecursion = [&](const auto& self, const ReshapeGroups& groups, std::size_t desiredIndex) -> std::optional<std::size_t> {
        const auto [trivialMerges, splits] = groups.count();
        if (
            trivialMerges > options.remainingMerges ||
            splits > options.remainingSplits
        ) {
            return std::nullopt;
        }
        if (trivialMerges + splits > options.overflow) {
            return Infinity; // Too many operations. Treat as inf.
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

} // namespace kas::ShapeComplexity
