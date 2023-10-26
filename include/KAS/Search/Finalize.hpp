#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "KAS/Core/FLOPsGame.hpp"
#include "KAS/Search/FinalStage.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/ShapeComplexity.hpp"
#include "KAS/Utils/Coroutine.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

// FLOPsGame that includes ShareOp's.
class ExtendedFLOPsGame {
    struct Adjacency {
        std::vector<std::size_t> increaseIndices;
        std::vector<std::size_t> decreaseIndices;
    };
    const BindingContext& ctx;
    Size inputSize;
    std::vector<Size> increase, decrease;
    std::vector<std::vector<bool>> dependencies;
    // Key: Share RHS, Iterator (which is also a special Expand). Basically what appears in weights.
    // Value: Adjacency. The indices of Unfolds and Expands that must be done before contraction, and the indices of reductions that must be done after contraction.
    std::map<Dimension, Adjacency, Dimension::AddressLessThan> sharedDependencies;
public:
    ExtendedFLOPsGame(const BindingContext& ctx, Size inputSize, const Graph& graph);
    // Look up adjacencies, and augment the dependencies. That is, during a contraction, all the dims in a single weight are added, so one decrease can depend on more increase's then explicitly derived.
    FLOPsGame getGameWithWeights(const std::vector<std::vector<Dimension>>& weights) const;
};

struct ColoredDimension {
    Dimension dim;
    WeightColor color;
    ColoredDimension(const Graph& graph, const Dimension& dim):
        dim { dim }, color { graph, dim } {}
    void removeAllRightTagsIn(const WeightColor& color) {
        this->color.removeAllRightTagsIn(color);
    }
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

    bool hasRedundantWeights(const Graph::DimensionSet& sharedWeightDims) const;

    static std::size_t MaxTensorsToMaxWeights(std::size_t maxTensors) { return maxTensors - 1; }

    struct FLOPsGameOptions {
        std::size_t maximumTensors;
        std::size_t maxFLOPs;
        // Required to be sorted by hash.
        const std::vector<ColoredDimension>& weightDims;
        // `selectedWeightDims` is required to be sorted by hash. So we can merge them efficiently.
        std::vector<ColoredDimension> buildFullWeightDims(const std::vector<ColoredDimension>& selectedWeightDims) const;
    };

    static std::size_t Distance(
        // Dimension and corresponding remainingLength, computed from maxChainLength.
        // Still required to be sorted.
        const std::vector<std::pair<Dimension, int>>& current,
        const Shape& desired, const Graph& graph, const ShapeComplexity::DistanceOptions& options, std::optional<FLOPsGameOptions> flopsOptions
    );

    KAS_STATISTICS_DEF(
        GenerateInvocations,
        SuccessfulInvocations,
        FailedInvocations,
        LegalFinalizations,
        UncanonicalWeight,
    )
    struct WeightOptions {
        std::size_t maxWeights;
        bool allowWeightPermutation;
    };
    using FinalStageBuilder = std::function<std::unique_ptr<FinalStage>(const FinalizeOp& op)>;
    struct GenerateOptions {
        const BindingContext& ctx;
        const Shape& desired;
        std::size_t maximumTensors;
        std::size_t maximumFinalizations;
        bool allowWeightPermutation;
        // For pruning.
        FinalStageBuilder finalStageBuilder;
        std::size_t maxFLOPs;
    };
    // If you need to disallow weight permutation, set maxHashFirstDimension.
    static Generator<std::vector<std::vector<Dimension>>> AssignToWeightsImpl(const std::vector<ColoredDimension>& remaining, std::size_t maxWeights, std::optional<std::size_t> maxHashFirstDimension);
    static Generator<std::vector<std::vector<Dimension>>> AssignToWeights(const std::vector<ColoredDimension>& weightDims, WeightOptions options);
    static std::vector<std::pair<FinalizeOp, std::unique_ptr<FinalStage>>> Generate(const GraphHandle& interface, const Graph& graph, const GenerateOptions& options);
};

struct NextFinalizeSlot: Next {
    FinalizeOp finalization;
    std::unique_ptr<FinalStage> nextStage;
    template<TopmostRange TR>
    static std::size_t GetKey(TR&& tensors) { return std::hash<std::vector<Topmost>>{}(tensors); }
    Arc toArc(const Sampler *sampler) const { return Arc(sampler, &finalization); }
};

} // namespace kas
