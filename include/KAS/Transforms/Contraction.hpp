#pragma once

#include "KAS/Core/Graph.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

enum class ContractionType: bool {
    Outer, // Basically a tensor product.
    Inner, // Basically a contraction.
};

class ContractionOp final: public Operation {
    bool isEqual(const Operation& other) const override;
public:
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
        ContractionType type() const noexcept;
        std::string description(const BindingContext& ctx) const;
        std::string descendantsDescription(const BindingContext& ctx) const;
    };
private:
    std::vector<Dimwise> dimwiseOps;
public:
    template<typename T>
    ContractionOp(T&& dimwiseOps): dimwiseOps(std::forward<T>(dimwiseOps)) {
        std::ranges::sort(this->dimwiseOps);
    }
    std::size_t opHash() const noexcept override;
    bool canApplyToInterface(const GraphHandle& interface) const override;
    void applyToInterface(GraphHandle& interface) const override;
    std::string description(const BindingContext& ctx) const override;
    std::string descendantsDescription(const BindingContext& ctx) const override;

    enum class SharedCandidateType {
        Normal, // Normal Share.
        Merge, // Merge input and weight, i.e., outer product.
        WeightsSharing, // Low-rank decomposition.
    };
    static SharedCandidateType GetSharedCandidateType(Dimension dim);
    struct CandidateDimension {
        Dimension dim;
        SharedCandidateType type;
        int lastWeight;
    };
    struct Analysis {
        // 0 means not contracted yet.
        int maxWeightId;
        // For canonicalization.
        std::optional<Dimension> lastWeightLeader;
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
        // Must be used with `full`.
        std::vector<SharedCandidateType> candidateTypes;
        // Number of total outer pruducts, i.e., Expand + Share.
        Size numelOuter;

        bool hasContracted() const { return maxWeightId > 0; }
    };
    static Analysis Analyze(const BindingContext& ctx, const Graph& graph, const GraphHandle& interface);
    struct GenerateOptions {
        const Analysis& analysis;
        const BindingContext& ctx;
        const Graph& graph;
        const Allowance& allowance;
        std::size_t maximumTensors;
        int maxShares;
        std::size_t maxExpansionMergeMultiplier;
        std::size_t maxExpansionWeightsSharingDimSize;
        std::size_t minExpansionWeightsSharingDimSize;
    };
    KAS_STATISTICS_DEF(
        GenerateInvocations,
        SuccessfulGenerations,
    );
    struct Enumerator {
        struct Options {
            OperationStore& store;
            const BindingContext& ctx;
            const Graph& graph;
            int weightId;
            std::optional<Dimension> lastWeightLeader;
            int maxShares;
            std::size_t maxExpansionMergeMultiplier;
            std::size_t maxExpansionWeightsSharingDimSize;
            std::size_t minExpansionWeightsSharingDimSize;
            const std::vector<CandidateDimension>& available;
        };
        const Options& options;
        const Allowance& allowance;
        const Size& numelOuter;
        std::size_t numShares;
        std::vector<std::optional<ContractionType>> assigned;
        Enumerator assign(std::optional<ContractionType> type, const Allowance& newAllowance, const Size& newNumelOuter) const;
        // Return nullptr if invalid.
        const ContractionOp *apply() const;
        Generator<const ContractionOp *> generate() const;
    };
    static std::vector<const ContractionOp *> Generate(OperationStore& store, const GenerateOptions& options);
};

static_assert(OperationImpl<ContractionOp>);

} // namespace kas
