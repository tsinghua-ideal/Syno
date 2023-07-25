#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest_prod.h>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"


namespace kas {

struct SampleOptions {
    using Seed = std::mt19937::result_type;
    Seed seed = 42;
    std::size_t depth = 10;
    std::size_t dimLowerBound = 1;
    std::size_t dimUpperBound = 8;
    std::size_t maximumTensors = 2;
    std::size_t maximumReductions = 2;
    std::size_t maxFLOPs = std::numeric_limits<std::size_t>::max();

    std::size_t maximumVariablesInSize = std::numeric_limits<std::size_t>::max();
    std::size_t maximumVariablesPowersInSize = std::numeric_limits<std::size_t>::max();
    bool requiresExactDivision = true;
    bool requiresOddKernelSizeInUnfold = true;

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
    float minimumMergeRatio = 2.0f;

    // Below are canonicalization options.

    // This option makes Split-Merge only be able to do views.
    bool disallowDiscontinuousView = false;

    // This option requires UnfoldOp chain to be in a specific order.
    bool canonicalizeUnfoldOrder = true;

    // Canonicalization rule set 1: at most one is true.
    bool disallowSplitRAboveUnfold = false;
    bool disallowUnfoldLAboveSplit = true;

    // Canonicalization rule set 2: at most one is true, but since this rule is too aggressive, we disable them by default.
    bool disallowMergeWithLargeBlockAboveUnfold = false;
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
        const Dimensions& interface;
        // Test hash equal.
        std::size_t hash;
    };

    struct Hash {
        using is_transparent = void;
        std::size_t operator()(const Query& query) const noexcept;
        std::size_t operator()(AbstractStage *stage) const noexcept;
    };
    struct Equal {
        using is_transparent = void;
        bool operator()(const Dimensions& lhs, const Dimensions& rhs) const noexcept;
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
    AbstractStage *find(std::size_t depth, const Dimensions& interface, std::unique_lock<std::recursive_mutex>& lock) const;
    AbstractStage *insert(std::size_t depth, std::unique_ptr<AbstractStage> stage, std::unique_lock<std::recursive_mutex>& lock);
    ~StageStore();
};

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
    Shape inputShape;
    Shape outputShape;

    std::vector<Iterator> outputIterators;
    std::vector<FixedDimension> fixedDimensions;
    Dimensions root;

    TensorExpression expressionOneTensor, expressionTwoTensors, expressionThreeTensors, expressionFourTensors;

    PrimitiveOpStore opStore;
    StageStore stageStore;

    ReductionStage *rootStage;

public:
    // A specification has the following forms:
    // <literal-value> [: <max-occurrencens>]
    // <variable-name> [= <literal-value>] [: <max-occurrencens>]
    Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const std::vector<std::map<std::string, std::size_t>>& allMappings, const std::vector<std::pair<std::size_t, std::size_t>>& fixedIODims, const SampleOptions& options = SampleOptions(), std::size_t numWorkerThreads = 1);
    Sampler(const Sampler&) = delete;
    Sampler(Sampler&&) = delete;

    BindingContext& getBindingContext() { return ctx; }
    const BindingContext& getBindingContext() const { return ctx; }
    Shape& getInputShape() { return inputShape; }
    Shape& getOutputShape() { return outputShape; }
    const SampleOptions& getOptions() const { return options; }
    PrimitiveOpStore& getOpStore() { return opStore; }
    StageStore& getStageStore() { return stageStore; }
    const Dimensions& getRootInterface() const { return root; }

    const std::vector<FixedDimension>& getFixedDimensions() const { return fixedDimensions; }
    const TensorExpression& getExpressionForTensorNum(std::size_t num) const;
    // Taking fixed dimensions into account.
    Size getTotalOutputSize() const;

    // The following APIs can be provided for Python bindings.
    std::optional<Node> visit(const std::vector<Next>& path);
    // The path is intended to visit a TensorView, but it may fail, in which case we rely on the search algorithm to penalize it.
    std::optional<std::pair<std::vector<Next>, Node>> randomNodeWithPrefix(const std::vector<Next>& prefix);
    // Visit multiple final nodes.
    std::vector<std::pair<std::vector<Next>, Node>> randomFinalNodesWithPrefix(const std::vector<Next>& prefix, std::size_t count);

    static void ConvertTensorViewToSearchableOrder(std::vector<std::vector<Dimension>>& tensorView);
    static std::vector<Next> ConvertGraphToPath(const Graph& graph);
    std::vector<Next> convertTensorsToPath(const std::vector<std::vector<Dimension>>& tensors) const;
    std::optional<std::vector<Arc>> convertPathToArcs(const std::vector<Next>& path);

    class Pruner {
        std::mutex mutex;
        std::condition_variable_any cv;
        std::queue<AbstractStage *> inbox;
        std::jthread thread;
        void handleUpdates(std::set<AbstractStage *>& updates);
    public:
        Pruner();
        void requestFinalizabilityUpdate(AbstractStage *stage, const AbstractStage *requestor);
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
public:
    std::recursive_mutex& getMutex(std::size_t depth, const Dimensions& interface);
    Pruner& getPruner() { return pruner; }
    Expander& getExpander() { return expander; }
};

} // namespace kas
