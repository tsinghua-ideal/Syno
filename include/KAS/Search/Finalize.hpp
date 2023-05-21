#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

struct FixedDimension;

// Contains multiple finalization options.
class FinalizeOp {
    friend class Stage;
    std::vector<Interface> tensors;
    std::size_t hash;

public:
    FinalizeOp(auto&& tensors):
        tensors { std::forward<decltype(tensors)>(tensors) }
    {
        std::size_t h = this->tensors.size();
        for (const auto& tensor : this->tensors) {
            HashCombine(h, tensor);
        }
        hash = h;
    }
    // Pass in sorted fixed dimensions.
    std::unique_ptr<TensorView> buildTensorView(const std::vector<FixedDimension>& fixed) const;
    std::size_t getHash() const noexcept { return hash; }

    std::string description(const BindingContext& ctx) const;

    static bool Prune(const std::vector<Graph::ConnectedComponent>& components, const std::vector<Interface>& trial);

    static inline std::size_t CountGenerateInvocations = 0;
    static inline std::size_t CountSuccessfulInvocations = 0;
    static inline std::size_t CountFailedInvocations = 0;
    static inline std::size_t CountLegalFinalizations = 0;
    static inline std::size_t CountConflictingColors = 0;
    static inline std::size_t CountPrunedFinalizations = 0;
    struct GenerateOptions {
        const BindingContext& ctx;
        const Shape& desired;
        std::size_t maximumTensors;
    };
    static std::vector<FinalizeOp> Generate(const ColoredInterface& interface, const Graph& graph, GenerateOptions options);
};

} // namespace kas
