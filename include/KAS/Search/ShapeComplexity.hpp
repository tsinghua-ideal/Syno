#pragma once

#include "KAS/Core/Size.hpp"
#include "KAS/Search/Common.hpp"


namespace kas::ShapeComplexity {

struct DistanceOptions {
    const BindingContext& ctx;
    int remainingMerges;
    int remainingSplits;
    int remainingUnfoldsAndExpands;
    std::size_t overflow;
};

class ReshapeGroup {
    Size remainder;
    bool hasNoInput;
    int splits;
    int merges;
    bool direct;
public:
    ReshapeGroup(const Size& provision, const Size& consumption);
    ReshapeGroup(const Size& provision);
    const Size& getRemainder() const;
    void addConsumption(const Size& consumption);
    void addProvision(const Size& provision);
    void markDirect();
    bool isDirect() const;
    bool isLegal() const;
    int countSplits() const;
    int countTrivialMerges() const;
    int countFinalAdditionalMerges() const;
    int countFinalUnfoldsAndExpands() const;
};

class Enumerator;

struct ReshapeGroups {
    static constexpr int NoGroup = -1;

    const Enumerator& enumerator;

    std::vector<ReshapeGroup> groups;

    std::vector<int> desiredToGroupId;
    std::vector<int> currentToGroupId;
    int vacantCurrents;

    const std::vector<DesiredSize>& desired() const;
    const std::vector<CurrentSize>& current() const;

    ReshapeGroups(const Enumerator& enumerator);

    void createGroup(std::size_t indexDesired, std::size_t indexCurrent);
    void createGroup(std::size_t indexCurrent);
    void addDesiredToGroup(std::size_t indexDesired, std::size_t indexGroup);
    void addCurrentToGroup(std::size_t indexCurrent, std::size_t indexGroup);

    bool desiredAssigned(std::size_t indexDesired) const;
    bool currentAssigned(std::size_t indexCurrent) const;
    int countGroups() const;
    int countVacantCurrents() const;
    struct Counts {
        int trivialMerges;
        int splits;
    };
    Counts count() const;

    Generator<ReshapeGroups> assignDesired(std::size_t indexDesired) const;
    Generator<ReshapeGroups> assignCurrent(std::size_t indexCurrent) const;
    struct CurrentCounts: Counts {
        int illegalGroups;
    };
    CurrentCounts currentCount() const;

    // Call only after assigning all sizes.
    bool isLegal() const;
    struct FinalCounts: Counts {
        int additionalMerges;
        int unfoldsAndExpands;
        int merges() const;
        std::size_t steps() const;
    };
    FinalCounts finalCount() const;
};

struct DeducedUnorderedDims {
    std::size_t indexDesired;
    std::set<int> unorderedCurrent;
};

class UnorderednessDeduction {
    // Keep this sorted.
    std::vector<DeducedUnorderedDims> content;
public:
    UnorderednessDeduction() = default;
    UnorderednessDeduction(const std::vector<DesiredSize>& desired, const std::vector<CurrentSize>& current);
    const std::vector<DeducedUnorderedDims>& get() const { return content; }
    bool noUnorderedDims() const { return content.empty(); }
    void accumulate(const ReshapeGroups& groups);
    void intersects(const UnorderednessDeduction& other);
};

constexpr std::size_t Infinity = std::numeric_limits<std::size_t>::max();

// Ideally, the groups are merge-split towers, i.e., reshapes.
// Let there be N groups. Then the waist is N sizes.
// N + #Splits = #current.
// N + #Merges = #desired + #Unfolds + #Expands.
// So basic corollaries are:
//  N >= #Unfolds + #Expands,
//  N >= #desired - #Merges,
//  N >= #current - #Splits.
// Now canonicalization prevents Merge above Split, but the counts remain the same.
class Enumerator {
public:
    const std::vector<DesiredSize>& desired;
    const std::vector<CurrentSize>& current;

private:
    DistanceOptions options;
    bool done = false;
    // If numels of desired and current are equal, then we do not need any Expand or Unfold.
    bool isExact = false;
    std::size_t bestSteps = Infinity;

    UnorderednessDeduction unorderedness;

    bool isCurrentDirect(std::size_t index) const;
    // If we can still deduce unordered dims, we need to traverse the whole space.
    // Otherwise we can truncate with best result.
    std::size_t overflow() const;
    void accumulateResult(const ReshapeGroups& groups, std::size_t steps);

    // First organize the desired sizes in groups.
    void matchDesired(const ReshapeGroups& groups, std::size_t desiredIndex);
    // Then check if the remaining current sizes can be matched to the groups.
    void matchCurrent(const ReshapeGroups& groups, std::size_t currentIndex);

public:
    Enumerator(const std::vector<DesiredSize>& desired, const std::vector<CurrentSize>& current, const DistanceOptions& options, const UnorderednessDeduction *unorderedness);
    void enumerate();
    std::size_t getBestSteps() const;
    const UnorderednessDeduction& getUnorderedness() const;
};

std::size_t Compute(const std::vector<DesiredSize>& desired, const std::vector<CurrentSize>& current, const DistanceOptions& options);

} // namespace kas::ShapeComplexity
