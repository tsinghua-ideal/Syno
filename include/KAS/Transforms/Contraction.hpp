#pragma once

#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

enum class ContractionType: bool {
    Outer, // Basically a tensor product.
    Inner, // Basically a contraction.
};

class ContractionOpStore;

class ContractionOp {
    struct Dimwise {
        // Non-null.
        const ShareOp *share;
        // Optional.
        // When there is ExpandOp, this is `i, j -> ij`, which can be outer product.
        // Otherwise, this is `i, i -> i`, which can be hadamard product or contraction.
        const ExpandOp *expand;
        bool operator==(const Dimwise& other) const noexcept = default;
        std::weak_ordering operator<=>(const Dimwise& other) const noexcept;
        std::size_t hash() const noexcept;
    };
    std::vector<Dimwise> dimwiseOps;
public:
    template<typename T>
    ContractionOp(T&& dimwiseOps): dimwiseOps(std::forward<T>(dimwiseOps)) {}
    bool operator==(const ContractionOp& other) const noexcept = default;
    std::size_t opHash() const noexcept;
    bool canApplyToInterface(const GraphHandle& interface) const;
    void applyToInterface(GraphHandle& interface) const;
    GraphHandle appliedToInterface(const GraphHandle& interface) const;
    std::string description(const BindingContext& ctx) const;
    std::string descendantsDescription(const BindingContext& ctx) const;
    struct Analysis {
        // 0 means not contracted yet.
        int maxWeightId;
        // Number of ShareOp's.
        int numShares;
        // Dims above the latest Share lhs if there is any contraction. If not, all the interface.
        // Note that we still exclude Share rhs.
        Topmost simpleViewSearchable;
        // Other parts of the interface.
        // Note that we still exclude Share rhs.
        Graph::DimensionSet other;
        // Only exclude Share rhs. SplitLikeOp may use this.
        Topmost full;

        bool hasContracted() const { return maxWeightId > 0; }
    };
    static Analysis Analyze(const GraphHandle& interface, const Graph& graph);
    struct GenerateOptions {
        const Analysis& analysis;
        const Graph& graph;
        const Allowance& allowance;
        std::size_t maximumTensors;
        int maxShares;
    };
    static std::vector<const ContractionOp *> Generate(PrimitiveOpStore& store, ContractionOpStore& contractionStore, const GenerateOptions& options);
};

static_assert(GeneralizedOp<ContractionOp>);

} // namespace kas
