#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Utils/Coroutine.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

struct FixedDimension;

// Contains multiple finalization options.
class FinalizeOp {
    friend class NormalStage;
    std::vector<std::vector<Dimension>> tensors;

public:
    FinalizeOp(auto&& tensors):
        tensors { std::forward<decltype(tensors)>(tensors) }
    {}
    // Pass in sorted fixed dimensions.
    std::shared_ptr<TensorView> buildTensorView(const std::vector<FixedDimension>& fixed, TensorExpression blending) const;
    std::size_t hash() const noexcept;

    std::string description(const BindingContext& ctx) const;

    static bool Prune(const std::vector<Graph::ConnectedComponent>& components, const std::vector<std::vector<Dimension>>& trial);

    struct WeightOptions {
        std::size_t maximumTensors;
    };

    static bool FitIntoWeights(const std::vector<std::reference_wrapper<const Dimension>>& current, const WeightOptions& options);

    struct DistanceOptions {
        const BindingContext& ctx;
        int remainingMerges;
        int remainingSplits;
        int remainingUnfolds;
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
        int countFinalUnfolds() const;
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
        int countTrivialMerges() const;
        int countSplits() const;

        Generator<ReshapeGroups> assignDesired(std::size_t indexDesired) const;
        Generator<ReshapeGroups> assignCurrent(std::size_t indexCurrent) const;
        int countIllegalGroups() const;

        // Call only after assigning all sizes.
        bool isLegal() const;
        int countFinalAdditionalMerges() const;
        int countFinalUnfolds() const;
        std::size_t countSteps() const;
    };

    static std::size_t ShapeComplexity(const Shape& desired, const std::vector<Size>& current, const FinalizeOp::DistanceOptions& options);

    static std::size_t Distance(const std::vector<std::reference_wrapper<const Dimension>>& current, const Shape& desired, const DistanceOptions& options);

    KAS_STATISTICS_DEF(
        GenerateInvocations,
        SuccessfulInvocations,
        FailedInvocations,
        LegalFinalizations,
        UncanonicalWeight,
        ConflictingColors,
        PrunedFinalizations,
    )
    struct GenerateOptions {
        const BindingContext& ctx;
        const Shape& desired;
        std::size_t maximumTensors;
    };

    static std::vector<FinalizeOp> Generate(const Dimensions& interface, const Graph& graph, const GenerateOptions& options);
};

struct NextFinalizeSlot: NextSlot<Next::Type::Finalize> {
    FinalizeOp finalization;
    template<TensorRange TR>
    static std::size_t GetKey(TR&& tensors) { return std::hash<std::vector<Dimensions>>{}(tensors); }
    Arc toArc() const { return Arc(&finalization); }
};

} // namespace kas
