#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "KAS/Search/FinalStage.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/ShapeComplexity.hpp"
#include "KAS/Utils/Coroutine.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

struct ColoredDimension {
    Dimension dim;
    WeightColor color;
    ColoredDimension(const Graph& graph, const Dimension& dim):
        dim { dim }, color { graph, dim } {}
    void removeAllRightTagsIn(const WeightColor& color) {
        this->color.removeAllRightTagsIn(color);
    }
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

    static std::size_t Distance(
        // Dimension and corresponding remainingLength, computed from maxChainLength.
        const std::vector<std::pair<Dimension, int>>& current,
        const Shape& desired, const Graph& graph, const ShapeComplexity::DistanceOptions& options
    );

    KAS_STATISTICS_DEF(
        GenerateInvocations,
        SuccessfulInvocations,
        FailedInvocations,
        LegalFinalizations,
        UncanonicalWeight,
    )
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

    static Generator<std::vector<std::vector<Dimension>>> AssignToWeights(const std::vector<ColoredDimension>& remaining, std::size_t maxWeights);
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
