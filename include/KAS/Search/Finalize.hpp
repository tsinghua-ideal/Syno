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
#include "KAS/Utils/Statistics.hpp"


namespace kas {

struct FixedDimension;

// Contains multiple finalization options.
class FinalizeOp {
    friend class Stage;
    std::vector<Interface> tensors;

public:
    FinalizeOp(auto&& tensors):
        tensors { std::forward<decltype(tensors)>(tensors) }
    {}
    // Pass in sorted fixed dimensions.
    std::shared_ptr<TensorView> buildTensorView(const std::vector<FixedDimension>& fixed) const;
    std::size_t hash() const noexcept;

    std::string description(const BindingContext& ctx) const;

    static bool Prune(const std::vector<Graph::ConnectedComponent>& components, const std::vector<Interface>& trial);

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
    static std::vector<FinalizeOp> Generate(const ColoredInterface& interface, const Graph& graph, const GenerateOptions& options);
};

struct NextFinalizeSlot: NextSlot<Next::Type::Finalize> {
    FinalizeOp finalization;
    template<TensorRange TR>
    static std::size_t GetKey(TR&& tensors) { return std::hash<std::vector<Interface>>{}(tensors); }
};

} // namespace kas
