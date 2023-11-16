#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "KAS/Search/FinalStage.hpp"
#include "KAS/Search/FLOPsGame.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/ShapeComplexity.hpp"
#include "KAS/Utils/Coroutine.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

struct ShapeDistance {
    std::size_t steps;
    std::size_t flops;
    bool operator==(const ShapeDistance& rhs) const noexcept = default;
    std::strong_ordering operator<=>(const ShapeDistance& rhs) const noexcept = default;
    static constexpr std::size_t MaxFLOPs = std::numeric_limits<std::size_t>::max();
    static const ShapeDistance Infinity;
    std::string toString() const;
};

struct ColoredDimension {
    Dimension dim;
    std::optional<int> weightId;
    ColoredDimension(const Dimension& dim, std::optional<int> weightId):
        dim { dim }, weightId { std::move(weightId) } {}
    std::size_t hash() const noexcept { return dim.hash(); }
};

struct FixedDimension;

// Contains multiple finalization options.
class FinalizeOp {
    friend class NormalStage;
    std::vector<Topmost> tensors;

public:
    FinalizeOp(auto&& tensors):
        tensors { std::forward<decltype(tensors)>(tensors) }
    {}
    // Pass in sorted fixed dimensions.
    TensorView buildTensorView(const std::vector<FixedDimension>& fixed, TensorExpression blending, const BindingContext& ctx) const;
    bool operator==(const FinalizeOp& rhs) const noexcept;
    std::size_t hash() const noexcept;
    std::size_t count() const noexcept { return tensors.size(); }

    GraphHandle toGraphHandle() const;

    double weightVariance(const ConcreteConsts& consts) const;
    double weightVariance(const BindingContext& ctx) const;

    std::string description(const BindingContext& ctx) const;

    static std::size_t MaxTensorsToMaxWeights(std::size_t maxTensors) { return maxTensors - 1; }

    struct FLOPsGameOptions {
        bool prune;
        std::size_t maximumTensors;
        std::size_t maxFLOPs;
        Size totalInputSize;
        // Required to be sorted by hash.
        const std::vector<ColoredDimension>& weightDims;
        // `selectedWeightDims` is required to be sorted by hash. So we can merge them efficiently.
        std::vector<ColoredDimension> buildFullWeightDims(const std::vector<ColoredDimension>& canBeWeightDims, const std::vector<bool>& select) const;
    };

    static ShapeDistance Distance(
        // Dimension and corresponding remainingLength, computed from maxChainLength.
        // Still required to be sorted.
        const std::vector<CurrentDimension>& current,
        const std::vector<DesiredSize>& desired,
        const Graph& graph, const ShapeComplexity::DistanceOptions& options, const FLOPsGameOptions& flopsOptions
    );

    KAS_STATISTICS_DEF(
        ShapeDistanceInvocations,
        ShapeDistanceTrials,
        ShapeDistanceTrialTooManySteps,
        ShapeDistanceTrialTooManyFLOPs,
        UnorderednessDeductionSuccess,
        UnorderednessDeductionFailure,
        ShapeDistanceUnorderedCanonicalized,
        GenerateInvocations,
        SuccessfulInvocations,
        FailedInvocations,
        UncanonicalUnorderedInput,
        CanonicalUnorderedInput,
        TooMuchPooling,
        LegalFinalizations,
        UncanonicalWeight,
    )
    using FinalStageBuilder = std::function<std::unique_ptr<FinalStage>(const FinalizeOp& op)>;
    struct GenerateOptions {
        const BindingContext& ctx;
        const std::vector<DesiredSize>& desired;
        std::size_t maximumTensors;
        std::size_t maximumFinalizations;
        bool allowWeightPermutation;
        std::size_t maxPoolingFactor;
        // For pruning.
        FinalStageBuilder finalStageBuilder;
        std::size_t maxFLOPs;
        std::size_t minFLOPs;
        std::size_t maxVRAM;
    };
    static std::vector<std::vector<Dimension>> AssignToWeights(const std::vector<ColoredDimension>& weightDims);
    static std::vector<std::pair<FinalizeOp, std::unique_ptr<FinalStage>>> Generate(const GraphHandle& interface, const Graph& graph, const GenerateOptions& options);
};

struct NextFinalizeSlot: Next {
    FinalizeOp finalization;
    std::unique_ptr<FinalStage> nextStage;
    template<TopmostRange TR>
    static std::size_t GetKey(TR&& tensors) { return std::hash<std::vector<Topmost>>{}(tensors); }
    Arc toArc(const Sampler *sampler) const { return Arc(sampler, &finalization); }
};

using NextFinalizeSlotStore = GenericNextSlotStore<NextFinalizeSlot>;

} // namespace kas
