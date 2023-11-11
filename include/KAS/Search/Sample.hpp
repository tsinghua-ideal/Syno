#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Search/Common.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Transforms/OperationStore.hpp"


namespace kas {

struct SampleOptions {
    using Seed = std::mt19937::result_type;
    Seed seed = 42;
    std::size_t depth = 10;
    std::size_t maxChainLength = 5;
    std::size_t maximumTensors = 2;
    std::size_t maximumReductions = 2;
    std::size_t maxFLOPs = std::numeric_limits<std::size_t>::max();
    std::size_t maxRDomSizeMultiplier = 32;

    bool enableFLOPsBasedPruning = true;

    std::size_t maximumEnumerationsPerVar = 5;
    std::size_t maximumVariablesInSize = std::numeric_limits<std::size_t>::max();
    std::size_t maximumVariablesPowersInSize = std::numeric_limits<std::size_t>::max();
    bool requiresExactDivision = true;
    bool requiresOddKernelSizeInUnfold = true;
    bool countCoefficientsInWeightsAsAllowanceUsage = false;

    std::string expressionOneTensor = "in_0";
    std::string expressionTwoTensors = "in_0 * in_1";
    std::string expressionThreeTensors = "in_0 * in_1 * in_2";
    std::string expressionFourTensors = "in_0 * in_1 * in_2 * in_3";
    std::size_t maximumFinalizations = 5;
    bool allowWeightPermutation = false;

    // These might better be the same.
    std::size_t maxStridedDimSize = 30;
    std::size_t maxUnfoldKernelSize = 30;

    float minimumUnfoldRatio = 2.0f;

    // If Split, Merge coincide with Shift, it is very likely that they are exchangeable.
    // Only if RHS of Split or Merge is comparatively small, Shift may make a difference when interchanged.
    float maximumValidReshapeShiftPattern = 5.0f;

    // ExpandOp related.
    bool disallowMergeInputAndWeight = false;
    bool disallowTile = true;
    bool disallowShareWeights = false;
    std::size_t maxExpansionRepeatMultiplier = 1;
    std::size_t maxExpansionMergeMultiplier = 512;
    std::size_t maxExpansionWeightsSharingDimSize = 8;
    std::size_t minExpansionWeightsSharingDimSize = 3;

    // Below are canonicalization options.

    // This option requires UnfoldOp chain to be in a specific order.
    bool canonicalizeUnfoldOrder = true;

    // Semantically we should do this, but in practice this prohibits optimizations.
    bool disallowSplitLAboveUnfold = false;

    // Canonicalization rule set 1: at most one is true.
    bool disallowSplitRAboveUnfold = false;
    bool disallowUnfoldLAboveSplit = true;

    // Canonicalization rule set 2: at most one is true, but since this rule is too aggressive, we disable them by default.
    // Time to get AGGRESSIVE!
    bool disallowMergeWithLargeBlockAboveUnfold = true;
    bool disallowUnfoldLAboveMergeR = false;

    // Canonicalization rule set 3: at most one is true. Since this rule perfectly preserves semantics, you'd better set exactly one of them to true.
    bool disallowSplitRAboveStride = false;
    bool disallowStrideAboveSplit = true;

    // Canonicalization rule set 4: at most one is true.
    bool disallowMergeWithLargeBlockAboveStride = true;
    bool disallowStrideAboveMergeR = false;

    // Canonicalization rule set 5: at most one is true.
    bool disallowUnfoldLAboveShift = true;
    bool disallowShiftAboveUnfold = false;

    // Some limit controls.
    int maximumExpands = -1;
    int maximumMerges = -1;
    int maximumSplits = -1;
    int maximumShifts = -1;
    int maximumStrides = -1;
    int maximumUnfolds = -1;
    int maximumShares = -1;

    void check() const;
};

static const SampleOptions DefaultSampleOptions = SampleOptions();

struct FixedDimension {
    std::size_t index;
    Dimension dim;
};

class StageStore {
    struct Query {
        // Test value equal.
        const GraphHandle& interface;
        // Test hash equal.
        std::size_t hash;
        Query(const GraphHandle& interface);
    };

    struct Hash {
        using is_transparent = void;
        std::size_t operator()(const Query& query) const noexcept;
        std::size_t operator()(const GraphHandle& interface) const noexcept;
        std::size_t operator()(AbstractStage *stage) const noexcept;
    };
    struct Equal {
        using is_transparent = void;
        bool operator()(const GraphHandle& lhs, const GraphHandle& rhs) const noexcept;
        bool operator()(const Query& lhs, const Query& rhs) const noexcept;
        bool operator()(const Query& lhs, AbstractStage *rhs) const noexcept;
        bool operator()(AbstractStage *lhs, const Query& rhs) const noexcept;
        bool operator()(AbstractStage *lhs, AbstractStage *rhs) const noexcept;
    };

private:
    const std::size_t bigBuckets;
    const std::size_t buckets;
    std::vector<std::vector<std::unordered_set<AbstractStage *, Hash, Equal>>> bucketsOfStages;

public:
    StageStore(std::size_t depth, std::size_t buckets):
        bigBuckets { depth },
        buckets { buckets },
        bucketsOfStages(bigBuckets)
    {
        for (auto& bucket : bucketsOfStages) {
            bucket.resize(buckets);
        }
    }
    AbstractStage *find(std::size_t depth, const GraphHandle& interface, std::unique_lock<std::recursive_mutex>& lock) const;
    AbstractStage *insert(std::size_t depth, std::unique_ptr<AbstractStage> stage, std::unique_lock<std::recursive_mutex>& lock);
    void forEach(auto&& f) {
        for (auto stage: bucketsOfStages | std::views::join | std::views::join) {
            std::invoke(std::forward<decltype(f)>(f), stage);
        }
    }
    ~StageStore();
};

struct MutexIndex;

class Sampler final {
    std::mt19937 rng;
    template<typename T>
    T random(T upper) {
        std::uniform_int_distribution<T> dist { 0, upper - 1 };
        return dist(rng);
    }
    const std::size_t numWorkerThreads;

    BindingContext ctx;
    SampleOptions options;
    Shape inputShape, outputShape;
    std::vector<Parser::Attributes> inputAttributes, outputAttributes;
    std::vector<DesiredSize> desiredShape;

    std::vector<Iterator> outputIterators;
    std::vector<FixedDimension> fixedDimensions;

    TensorExpression expressionOneTensor, expressionTwoTensors, expressionThreeTensors, expressionFourTensors;

    OperationStore opStore;
    StageStore stageStore;

    ReductionStage *rootStage;

    std::array<DepthwiseStatistics, DepthwiseStatistics::MaxSearchDepth> depthwiseStatistics {};

public:
    // A specification has the following forms:
    // <literal-value> [: <max-occurrencens>]
    // <variable-name> [= <literal-value>] [: <max-occurrencens>]
    Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const std::vector<std::map<std::string, std::size_t>>& allMappings, const std::vector<std::pair<std::size_t, std::size_t>>& fixedIODims, const SampleOptions& options = SampleOptions(), std::size_t numWorkerThreads = 1);
    Sampler(const Sampler&) = delete;
    Sampler(Sampler&&) = delete;

    const BindingContext& getBindingContext() const { return ctx; }
    const Shape& getInputShape() const { return inputShape; }
    // Search-time. That is, all fixed dimensions are removed.
    const std::vector<DesiredSize>& getDesiredShape() const { return desiredShape; }
    const Shape& getOutputShape() const { return outputShape; }
    const std::vector<Parser::Attributes>& getInputAttributes() const { return inputAttributes; }
    const std::vector<Parser::Attributes>& getOutputAttributes() const { return outputAttributes; }
    const SampleOptions& getOptions() const { return options; }
    OperationStore& getOpStore() { return opStore; }
    StageStore& getStageStore() { return stageStore; }
    DepthwiseStatistics& getStats(std::size_t depth) { return depthwiseStatistics[depth]; }
    std::size_t getExpandAtDepth();

    std::string statsToString() const;

    const std::vector<FixedDimension>& getFixedDimensions() const { return fixedDimensions; }
    const TensorExpression& getExpressionForTensorNum(std::size_t num) const;
    // The size of fixed dimensions.
    Size getFixedDimensionsSize() const;
    // Taking fixed dimensions into account.
    Size getTotalInputSize() const;
    // Taking fixed dimensions into account.
    Size getTotalOutputSize() const;
    Size getMaxRDomSize() const;
    int remainingChainLength(const Graph& graph, const Dimension& dim) const;

    // The following APIs can be provided for Python bindings.
    std::optional<Node> visit(const std::vector<Next>& path);
    // The path is intended to visit a FinalStage, but it may fail, in which case we rely on the search algorithm to penalize it.
    std::optional<std::pair<std::vector<Next>, Node>> randomNodeWithPrefix(const std::vector<Next>& prefix);
    // Visit multiple final nodes.
    std::vector<std::pair<std::vector<Next>, Node>>
    randomFinalNodesWithPrefix(const std::vector<Next> &prefix, std::size_t count, std::optional<Next::Type> type = std::nullopt, int steps = 1);

    // Remove all the fixed dimensions.
    void removeFixedDimensions(std::vector<Topmost>& tensors) const;
    // Sort the dimensions of weights in order of hash. This also sorts the weights if permutation of weights is disallowed.
    void sortAllExpansionsAndWeightDimensions(std::vector<Topmost>& tensors) const;
    // This calls the above two functions. Then you can use these tensors to build a GraphHandle.
    void convertTensorsToSearchableForm(std::vector<Topmost>& tensors) const;
    // A convenience method.
    std::vector<Topmost> convertTensorViewToSearchableTensors(const TensorView& tensorView) const;
    static Next ConvertSearchableTensorsToFinalNext(const std::vector<Topmost>& tensors);

    // This cannot figure out Finalize.
    static std::vector<const Operation *> ConvertGraphToOps(const Graph& graph);
    static std::vector<Next> ConvertOpsToNexts(const std::vector<const Operation *>& ops);
    std::vector<Arc> convertOpsToArcs(const std::vector<const Operation *>& ops) const;

    // For Forward.
    static std::vector<Next> ConvertSearchableTensorsToPath(const std::vector<Topmost>& tensors);
    // Convenience.
    std::vector<Next> convertTensorViewToPath(const TensorView& tensorView) const;

    class Pruner {
        std::mutex mutex;
        std::condition_variable_any cv;
        std::queue<AbstractStage *> inbox;
        std::jthread thread;
        std::size_t committed = 0;
        std::size_t completed = 0;
        bool clear() { return committed == completed; }
        std::condition_variable cvCompleted;
        void handleUpdates(std::set<AbstractStage *>& updates);
    public:
        Pruner();
        void requestFinalizabilityUpdate(AbstractStage *stage, const AbstractStage *requestor);
        void sync();
        ~Pruner();
    };
    class Expander {
        struct Task {
            Node node;
            int layers;
        };
        // A thread pool that accepts the tasks.
        std::vector<std::jthread> threads;
        // A queue of tasks.
        std::mutex mutex;
        std::condition_variable_any cv;
        std::queue<Task> inbox;
        std::size_t submitted = 0;
        std::size_t finished = 0;
        bool ready = true;
        std::condition_variable cvReady;
        void finish();
    public:
        Expander(std::size_t numThreads);
        void expand(Node node, int layers);
        void expandSync(Node node, int layers);
        void sync();
        ~Expander();
    };
private:
    const std::size_t countMutexesInLayer;
    std::vector<std::vector<std::recursive_mutex>> mutexes;
    Pruner pruner;
    Expander expander;
    ThreadPool<LatticeTask> latticeExpander;
public:
    std::recursive_mutex& getMutex(MutexIndex index);
    Pruner& getPruner() { return pruner; }
    Expander& getExpander() { return expander; }
    ThreadPool<LatticeTask>& getLatticeExpander() { return latticeExpander; }
};

} // namespace kas
