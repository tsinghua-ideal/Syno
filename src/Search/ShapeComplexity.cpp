#include "KAS/Search/ShapeComplexity.hpp"


namespace kas::ShapeComplexity {

ReshapeGroup::ReshapeGroup(const Size& provision, const Size& consumption):
    remainder { provision / consumption },
    hasNoInput { false },
    splits { 0 },
    merges { 0 },
    direct { false }
{}

ReshapeGroup::ReshapeGroup(const Size& provision):
    remainder { provision },
    hasNoInput { true },
    splits { 0 },
    merges { 0 },
    direct { false }
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
    remainder /= consumption;
}

void ReshapeGroup::addProvision(const Size& provision) {
    ++splits;
    remainder *= provision;
}

void ReshapeGroup::markDirect() {
    KAS_ASSERT(!direct);
    direct = true;
}

bool ReshapeGroup::isDirect() const {
    return direct;
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

int ReshapeGroup::countFinalUnfoldsAndExpands() const {
    if (hasNoInput) {
        return 1;
    }
    auto trait = remainder.getTrait();
    return static_cast<int>(trait.value() != Size::Trait::One);
}

const std::vector<DesiredSize>& ReshapeGroups::desired() const {
    return enumerator.desired;
}
const std::vector<CurrentSize>& ReshapeGroups::current() const {
    return enumerator.current;
}

ReshapeGroups::ReshapeGroups(const Enumerator& enumerator):
    enumerator { enumerator },
    desiredToGroupId(enumerator.desired.size(), NoGroup),
    currentToGroupId(enumerator.current.size(), NoGroup),
    vacantCurrents { static_cast<int>(enumerator.current.size()) }
{}

void ReshapeGroups::createGroup(std::size_t indexDesired, std::size_t indexCurrent) {
    KAS_ASSERT(!desiredAssigned(indexDesired) && !currentAssigned(indexCurrent));
    desiredToGroupId[indexDesired] = groups.size();
    currentToGroupId[indexCurrent] = groups.size();
    --vacantCurrents;
    groups.emplace_back(current()[indexCurrent], desired()[indexDesired]);
}

void ReshapeGroups::createGroup(std::size_t indexCurrent) {
    KAS_ASSERT(!currentAssigned(indexCurrent));
    currentToGroupId[indexCurrent] = groups.size();
    --vacantCurrents;
    groups.emplace_back(current()[indexCurrent]);
}

void ReshapeGroups::addDesiredToGroup(std::size_t indexDesired, std::size_t indexGroup) {
    KAS_ASSERT(!desiredAssigned(indexDesired) && countGroups() > indexGroup);
    desiredToGroupId[indexDesired] = indexGroup;
    auto& group = groups[indexGroup];
    group.addConsumption(desired()[indexDesired]);
}

void ReshapeGroups::addCurrentToGroup(std::size_t indexCurrent, std::size_t indexGroup) {
    KAS_ASSERT(!currentAssigned(indexCurrent) && countGroups() > indexGroup);
    currentToGroupId[indexCurrent] = indexGroup;
    --vacantCurrents;
    auto& group = groups[indexGroup];
    KAS_ASSERT(current()[indexCurrent].remainingLength > 0, "Cannot add a current size with remainingLength == 0 to a group!");
    group.addProvision(current()[indexCurrent]);
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
    const Size& desiredSize = desired()[indexDesired];
    KAS_ASSERT(Size::GetLimitsUsage(desiredSize.getPrimary()).varsPowersInSize == 1, "Input dimension sizes must each contain one and only one primary variable! TODO: support other shapes.");
    std::size_t varId = std::numeric_limits<std::size_t>::max();
    for (std::size_t pId = 0; auto p: desiredSize.getPrimary()) {
        if (p == 1) {
            varId = pId;
        }
        ++pId;
    }

    // Decide which group this desired dimension should join.
    // First is existing groups.
    for (std::size_t i = 0; i <= countGroups(); ++i) {
        const bool isNewGroup = i == countGroups();
        if (!isNewGroup && groups[i].isDirect()) continue;
        if (isNewGroup || groups[i].getRemainder().getPrimary()[varId] == 0) {
            // Check new sizes for this variable.
            for (std::size_t j = 0; j < current().size(); ++j) {
                if (currentAssigned(j)) continue;
                const auto& [currentSize, currentRemainingLength] = current()[j];
                if (currentSize.getPrimary()[varId] > 0) {
                    if (currentRemainingLength == 0) {
                        // We cannot allow for another Op.
                        // So this group must be exact! That is, one input and one output.
                        if (!isNewGroup || desiredSize != currentSize) {
                            continue;
                        }
                    }
                    auto copy = *this;
                    if (isNewGroup) {
                        copy.createGroup(indexDesired, j);
                        if (currentRemainingLength == 0) copy.groups[i].markDirect();
                    } else {
                        copy.addCurrentToGroup(j, i);
                        copy.addDesiredToGroup(indexDesired, i);
                    }
                    co_yield std::move(copy);
                }
            }
        } else {
            // This group provides this variable. OK.
            auto copy = *this;
            copy.addDesiredToGroup(indexDesired, i);
            co_yield std::move(copy);
        }
    }
    co_return;
}

auto ReshapeGroups::assignCurrent(std::size_t indexCurrent) const -> Generator<ReshapeGroups> {
    // Only if the current size has remainingLength.
    // This is because we need another Op to eliminate it in any group.
    if (current()[indexCurrent].remainingLength > 0) {
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
    }
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
        counts.unfoldsAndExpands += group.countFinalUnfoldsAndExpands();
    }
    return counts;
}

int ReshapeGroups::FinalCounts::merges() const {
    return trivialMerges + additionalMerges;
}

std::size_t ReshapeGroups::FinalCounts::steps() const {
    return trivialMerges + splits + additionalMerges + unfoldsAndExpands;
}

UnorderednessDeduction::UnorderednessDeduction(const std::vector<DesiredSize>& desired, const std::vector<CurrentSize>& current) {
    std::set<int> currentIndices; // 0..<current.size()
    std::ranges::copy(std::views::iota(0, static_cast<int>(current.size())), std::inserter(currentIndices, currentIndices.begin()));
    for (std::size_t i = 0; const auto& desiredSize: desired) {
        if (desiredSize.isUnordered) {
            content.emplace_back(i, currentIndices);
        }
        ++i;
    }
}

void UnorderednessDeduction::accumulate(const ReshapeGroups& groups) {
    // A fast path. If we are already sure we cannot extract any information, we can skip this check.
    // More importantly, we can update the overflow according to bestSteps.
    if (noUnorderedDims()) return;
    decltype(content) newContent;
    for (auto& deduction: content) {
        auto gid = groups.desiredToGroupId[deduction.indexDesired];
        // We can only deduce if the group has a unique input, which is, this unordered input.
        if (std::ranges::count_if(groups.desiredToGroupId, [gid](int g) { return g == gid; }) != 1) {
            continue;
        }
        auto newDeduction = DeducedUnorderedDims { deduction.indexDesired, {} };
        // Take the intersection.
        for (std::size_t i = 0; i < groups.currentToGroupId.size(); ++i) {
            if (
                groups.currentToGroupId[i] == gid
                && deduction.unorderedCurrent.contains(i)
            ) {
                newDeduction.unorderedCurrent.emplace(i);
            }
        }
        if (newDeduction.unorderedCurrent.empty()) {
            // We have deduced that there is nothing to be determined.
            continue;
        }
        newContent.emplace_back(std::move(newDeduction));
    }
    content = std::move(newContent);
}

void UnorderednessDeduction::intersects(const UnorderednessDeduction& other) {
    const auto& lhs = content;
    const auto& rhs = other.content;
    auto it1 = lhs.begin();
    auto it2 = rhs.begin();
    decltype(content) newContent;
    while (it1 != lhs.end() && it2 != rhs.end()) {
        if (it1->indexDesired < it2->indexDesired) {
            ++it1;
        } else if (it1->indexDesired > it2->indexDesired) {
            ++it2;
        } else {
            // Same indexDesired.
            std::set<int> intersection;
            std::ranges::set_intersection(it1->unorderedCurrent, it2->unorderedCurrent, std::inserter(intersection, intersection.begin()));
            if (!intersection.empty()) {
                newContent.emplace_back(it1->indexDesired, std::move(intersection));
            }
            ++it1;
            ++it2;
        }
    }
    content = std::move(newContent);
}

bool Enumerator::isCurrentDirect(std::size_t index) const {
    return current[index].remainingLength == 0;
}

std::size_t Enumerator::overflow() const {
    if (unorderedness.noUnorderedDims()) {
        return std::min(bestSteps, options.overflow);
    } else {
        return options.overflow;
    }
}

void Enumerator::accumulateResult(const ReshapeGroups& groups, std::size_t steps) {
    bestSteps = std::min(bestSteps, steps);
    // By the enumeration we can determine the unordered current dimensions.
    // We can use this to do pruning.
    unorderedness.accumulate(groups);
}

void Enumerator::matchDesired(const ReshapeGroups& groups, std::size_t desiredIndex) {
    const auto [trivialMerges, splits] = groups.count();
    if (
        // Exceeds limits.
        trivialMerges > options.remainingMerges ||
        splits > options.remainingSplits ||
        // Too many operations.
        trivialMerges + splits > overflow()
    ) {
        return;
    }
    if (desiredIndex != desired.size()) {
        for (ReshapeGroups newGroups: groups.assignDesired(desiredIndex)) {
            matchDesired(newGroups, desiredIndex + 1);
        }
    } else {
        // We have assigned all desired sizes.
        // Check if there are current dimensions that have no remainingLength but has not been assigned.
        for (std::size_t i = 0; i < current.size(); ++i) {
            if (!groups.currentAssigned(i) && isCurrentDirect(i)) {
                return;
            }
        }
        matchCurrent(groups, 0);
    }
}

void Enumerator::matchCurrent(const ReshapeGroups& groups, std::size_t currentIndex) {
    const auto counts = groups.currentCount();
    const int trivialMerges = counts.trivialMerges;
    const int splits = counts.splits;
    const int illegalGroups = counts.illegalGroups;
    if (
        // Exceeds limits.
        trivialMerges > options.remainingMerges ||
        splits > options.remainingSplits ||
        // Need at least one current size to make a group legal.
        illegalGroups > groups.countVacantCurrents() ||
        // Note that in this stage, each vacant current size accounts for at least one step.
        trivialMerges + splits + groups.countVacantCurrents() > overflow()
    ) {
        return;
    }
    if (currentIndex != current.size()) {
        if (groups.currentAssigned(currentIndex)) {
            matchCurrent(groups, currentIndex + 1);
        } else {
            for (ReshapeGroups newGroups: groups.assignCurrent(currentIndex)) {
                matchCurrent(newGroups, currentIndex + 1);
            }
        }
    } else {
        // We have grouped all sizes.
        // We do not want any illegal sizes.
        if (!groups.isLegal()) {
            return;
        }
        const auto finalCounts = groups.finalCount();
        if (
            finalCounts.unfoldsAndExpands > options.remainingUnfoldsAndExpands ||
            finalCounts.merges() > options.remainingMerges
        ) {
            return;
        }
        const std::size_t steps = finalCounts.steps();
        if (steps > overflow()) {
            return;
        }
        accumulateResult(groups, steps);
    }
}

Enumerator::Enumerator(const std::vector<DesiredSize>& desired, const std::vector<CurrentSize>& current, const DistanceOptions& options, const UnorderednessDeduction *unorderedness):
    desired(desired),
    current(current),
    options { options }
{
    KAS_ASSERT(!desired.empty());

    // First, check whether there are enough elements in the input tensor.
    if (current.empty()) {
        done = true;
        return;
    }
    Size desiredElements = Size::Product(desired | std::views::transform(&DesiredSize::value)); // desired numel.
    Size quotient = Size::Product(current | std::views::transform(&CurrentSize::value)); // current numel.
    auto quotientSpec = quotient.testDividedBy(desiredElements);
    if (
        !quotientSpec
        || *quotientSpec == Size::Trait::IllegalCoefficient
        || (options.ctx.requiresExactDivision() && quotient.lowerBoundEst(options.ctx) < 1_uz)
    ) {
        // Not enough elements.
        done = true;
        return;
    }
    if (*quotientSpec == Size::Trait::One) {
        // If this is exact, we do not need any Expand or Unfold.
        isExact = true;
    }

    // If this is exact, we can deduce unordered groups.
    if (isExact) {
        if (unorderedness) {
            this->unorderedness = *unorderedness;
        } else {
            this->unorderedness = UnorderednessDeduction { desired, current };
        }
    }

    return;
}

void Enumerator::enumerate() {
    if (!done) {
        // Group the dimensions recursively.
        matchDesired(ReshapeGroups { *this }, 0);
        done = true;
    }
}

std::size_t Enumerator::getBestSteps() const {
    KAS_ASSERT(done);
    return bestSteps;
}

const UnorderednessDeduction& Enumerator::getUnorderedness() const {
    KAS_ASSERT(done);
    return unorderedness;
}

std::size_t Compute(const std::vector<DesiredSize>& desired, const std::vector<CurrentSize>& current, const DistanceOptions& options) {
    Enumerator enumerator { desired, current, options, nullptr };
    enumerator.enumerate();
    return enumerator.getBestSteps();
}

} // namespace kas::ShapeComplexity
