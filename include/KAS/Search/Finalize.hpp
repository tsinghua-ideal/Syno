#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/ShapeComplexity.hpp"
#include "KAS/Utils/Coroutine.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

struct ColoredDimension {
    Dimension dim;
    WeightColor color;
    ColoredDimension(Dimension dim): dim { dim }, color { dim } {}
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
    std::shared_ptr<TensorView> buildTensorView(const std::vector<FixedDimension>& fixed, TensorExpression blending) const;
    bool operator==(const FinalizeOp& rhs) const noexcept;
    std::size_t hash() const noexcept;
    std::size_t count() const noexcept { return tensors.size(); }

    // TODO!!!
    GraphHandle toGraphHandle() const;

    double weightVariance(const ConcreteConsts& consts) const;
    double weightVariance(const BindingContext& ctx) const;

    std::string description(const BindingContext& ctx) const;

    static bool Prune(const std::vector<Graph::ConnectedComponent>& components, const std::vector<std::vector<Dimension>>& trial);

    struct WeightOptions {
        std::size_t maximumTensors;
    };

    static bool FitIntoWeights(const std::vector<Dimension>& current, const WeightOptions& options);

    static std::size_t Distance(const std::vector<Dimension>& current, const Shape& desired, const ShapeComplexity::DistanceOptions& options);

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
        std::size_t maximumFinalizations;
        bool allowWeightPermutation;
    };

    static Generator<std::vector<std::vector<Dimension>>> AssignToWeights(const std::vector<ColoredDimension>& remaining, std::size_t maxWeights);
    static std::vector<FinalizeOp> Generate(const GraphHandle& interface, const Graph& graph, const GenerateOptions& options);
};

struct NextFinalizeSlot: Next {
    FinalizeOp finalization;
    template<TensorRange TR>
    static std::size_t GetKey(TR&& tensors) { return std::hash<std::vector<std::vector<Dimension>>>{}(tensors); }
    Arc toArc(const Sampler *sampler) const { return Arc(sampler, &finalization); }
};

} // namespace kas
