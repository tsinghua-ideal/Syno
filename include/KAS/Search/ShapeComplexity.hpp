#pragma once

#include "KAS/Core/Size.hpp"


namespace kas::ShapeComplexity {

struct DistanceOptions {
    const BindingContext& ctx;
    int remainingMerges;
    int remainingSplits;
    int remainingUnfolds;
    int remainingExpands;
    int remainingUnfoldsAndExpands() const {
        return remainingUnfolds + remainingExpands;
    }
    std::size_t overflow;
};

class ReshapeGroup {
    Size remainder;
    bool hasNoInput;
    int splits;
    int merges;
public:
    ReshapeGroup(const Size& provision, const Size& consumption);
    ReshapeGroup(const Size& provision);
    const Size& getRemainder() const;
    void addConsumption(const Size& consumption);
    void addProvision(const Size& provision);
    bool isLegal() const;
    int countSplits() const;
    int countTrivialMerges() const;
    int countFinalAdditionalMerges() const;
    int countFinalUnfoldsAndExpands() const;
};

struct ReshapeGroups {
    static constexpr int NoGroup = -1;

    const Shape& desired;
    const std::vector<Size>& current;

    std::vector<ReshapeGroup> groups;

    std::vector<int> desiredToGroupId;
    std::vector<int> currentToGroupId;
    int vacantCurrents;

    ReshapeGroups(const Shape& desired, const std::vector<Size>& current);

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

constexpr std::size_t Infinity = std::numeric_limits<std::size_t>::max();

std::size_t Compute(const Shape& desired, const std::vector<Size>& current, const DistanceOptions& options);

} // namespace kas::ShapeComplexity
