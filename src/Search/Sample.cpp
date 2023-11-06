#include <future>
#include <iterator>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/Parser.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/TensorView.hpp"
#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Search/NormalStage.hpp"
#include "KAS/Search/ReductionStage.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Threads.hpp"


namespace kas {

void SampleOptions::check() const {
    KAS_ASSERT(dimLowerBound >= 1);
    KAS_ASSERT(dimUpperBound >= dimLowerBound);
    KAS_ASSERT(maximumTensors >= 1);
    KAS_ASSERT(maximumTensors <= 4);
    KAS_ASSERT(maxStridedDimSize > 1);
    KAS_ASSERT(maxUnfoldKernelSize > 1);
    KAS_ASSERT(minimumUnfoldRatio >= 1.0f);
    KAS_ASSERT(!requiresOddKernelSizeInUnfold || requiresExactDivision, "requiresOddKernelSizeInUnfold requires requiresExactDivision.");
    KAS_ASSERT(disallowSplitRAboveUnfold + disallowUnfoldLAboveSplit <= 1);
    KAS_ASSERT(disallowMergeWithLargeBlockAboveUnfold + disallowUnfoldLAboveMergeR <= 1);
    KAS_ASSERT(disallowSplitRAboveStride + disallowStrideAboveSplit <= 1);
    KAS_ASSERT(disallowMergeWithLargeBlockAboveStride + disallowStrideAboveMergeR <= 1);
    KAS_ASSERT(disallowUnfoldLAboveShift + disallowShiftAboveUnfold <= 1);
}

StageStore::Query::Query(const GraphHandle& interface):
    interface { interface },
    hash { Hash{}(interface) }
{}

std::size_t StageStore::Hash::operator()(const Query& query) const noexcept {
    return query.hash;
}
std::size_t StageStore::Hash::operator()(const GraphHandle& interface) const noexcept {
    return interface.hash();
}
std::size_t StageStore::Hash::operator()(AbstractStage *stage) const noexcept {
    return (*this)(stage->getInterface());
}

bool StageStore::Equal::operator()(const GraphHandle& lhs, const GraphHandle& rhs) const noexcept {
    return lhs == rhs;
}
bool StageStore::Equal::operator()(const Query& lhs, const Query& rhs) const noexcept {
    return (*this)(lhs.interface, rhs.interface);
}
bool StageStore::Equal::operator()(const Query& lhs, AbstractStage *rhs) const noexcept {
    return (*this)(lhs.interface, rhs->getInterface());
}
bool StageStore::Equal::operator()(AbstractStage *lhs, const Query& rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs.interface);
}
bool StageStore::Equal::operator()(AbstractStage *lhs, AbstractStage *rhs) const noexcept {
    return (*this)(lhs->getInterface(), rhs->getInterface());
}

AbstractStage *StageStore::find(std::size_t depth, const GraphHandle& interface, std::unique_lock<std::recursive_mutex>& lock) const {
    KAS_ASSERT(lock.owns_lock());
    auto q = Query(interface);
    const auto& bucket = bucketsOfStages[depth][q.hash % buckets];
    if (auto it = bucket.find(q); it != bucket.end()) {
        return *it;
    } else {
        return nullptr;
    }
}

AbstractStage *StageStore::insert(std::size_t depth, std::unique_ptr<AbstractStage> stage, std::unique_lock<std::recursive_mutex>& lock) {
    KAS_ASSERT(lock.owns_lock());
    std::size_t bucketIndex = Hash{}(stage->getInterface()) % buckets;
    auto& bucket = bucketsOfStages[depth][bucketIndex];
    auto [it, inserted] = bucket.insert(stage.get());
    if (inserted) {
        return stage.release();
    }
    return nullptr;
}

StageStore::~StageStore() {
    for (auto stage: bucketsOfStages | std::views::join | std::views::join) {
        delete stage;
    }
}

namespace {

// By birthday paradox, about 1/8 probability of collision.
std::size_t MutexCountFromNumWorkers(std::size_t numWorkerThreads) {
    return 4 * numWorkerThreads * numWorkerThreads;
}

}

Sampler::Sampler(std::string_view inputShape, std::string_view outputShape, const std::vector<std::string>& primarySpecs, const std::vector<std::string>& coefficientSpecs, const std::vector<std::map<std::string, std::size_t>>& allMappings, const std::vector<std::pair<std::size_t, std::size_t>>& fixedIODims, const SampleOptions& options, std::size_t numWorkerThreads):
    rng { options.seed },
    numWorkerThreads { numWorkerThreads },
    options { options },
    stageStore { options.depth + 1, MutexCountFromNumWorkers(numWorkerThreads) },
    countMutexesInLayer { MutexCountFromNumWorkers(numWorkerThreads) },
    mutexes(options.depth + 1),
    pruner {},
    expander { numWorkerThreads },
    latticeExpander { numWorkerThreads, [](ThreadPool<LatticeTask>& expander, LatticeTask task) {
        task.node.expandWithArcs(expander, task);
    } }
{
    this->options.check();

    KAS_ASSERT(numWorkerThreads > 0);
    for (auto& mutexes: this->mutexes) {
        std::vector<std::recursive_mutex> v(countMutexesInLayer);
        mutexes = std::move(v);
    }

    // First parse the variable names in specifications. Unnamed variables are named by x_i and c_i.
    auto parser = ShapeSpecParser(primarySpecs, coefficientSpecs);

    // Then we collect the variables in in/out shapes. In case of new variable, add it to primary.
    parser.addShape(inputShape).addShape(outputShape);

    // Put the specs in order.
    auto [contractedPrimarySpecs, contractedCoefficientSpecs] = parser.build();

    // Apply the specs to all variables, and the mappings to obtain concrete consts.
    ctx = BindingContext(contractedPrimarySpecs, contractedCoefficientSpecs, allMappings, {
        .maximumEnumerationsPerVar = options.maximumEnumerationsPerVar,
        .maximumVariablesInSize = options.maximumVariablesInSize,
        .maximumVariablesPowersInSize = options.maximumVariablesPowersInSize,
        .requiresExactDivision = options.requiresExactDivision,
    });

    // Parse shape from names.
    std::tie(this->inputShape, inputAttributes) = ctx.getShapeAndAttributes(inputShape);
    std::tie(this->outputShape, outputAttributes) = ctx.getShapeAndAttributes(outputShape);

    // Input unorderedness.
    for (std::size_t index = 0; const auto& attrs: inputAttributes) {
        desiredShape.emplace_back(this->inputShape[index], attrs.contains("unordered"));
        ++index;
    }
    KAS_ASSERT(desiredShape.size() == this->inputShape.size());

    // The root, which are output iterators.
    std::vector<Dimension> rootDimensions;
    // Initialize the output iterators.
    outputIterators.reserve(this->outputShape.size());
    for (std::size_t index = 0; const auto& domain: this->outputShape) {
        outputIterators.emplace_back(index, domain, outputAttributes.at(index).contains("unordered"));
        rootDimensions.emplace_back(&outputIterators.back()); // We have called reserve in advance so this is safe.
        ++index;
    }

    // Check that the bound I/O dimensions are of the same size, and collect fixedDimensions.
    std::vector<std::size_t> boundOutputDimensions;
    for (auto [i, o]: fixedIODims) {
        KAS_ASSERT(this->inputShape[i] == this->outputShape[o], "Bound I/O dimensions must be of the same size.");
        fixedDimensions.emplace_back(i, &outputIterators[i]);
        boundOutputDimensions.emplace_back(o);
    }
    // Sort fixedDimensions by input index. We will insert them back into tensors in FinalizeOp::buildTensorView().
    std::ranges::sort(fixedDimensions, std::less{}, &FixedDimension::index);
    std::ranges::sort(boundOutputDimensions);
    // To bind input and output dimensions, remove fixed dimensions from inputShape and outputShape.
    for (std::size_t i: fixedDimensions | std::views::reverse | std::views::transform(&FixedDimension::index)) {
        this->inputShape.sizes.erase(this->inputShape.sizes.begin() + i);
        this->desiredShape.erase(this->desiredShape.begin() + i);
    }
    for (std::size_t o: boundOutputDimensions | std::views::reverse) {
        this->outputShape.sizes.erase(this->outputShape.sizes.begin() + o);
        rootDimensions.erase(rootDimensions.begin() + o);
    }

    expressionOneTensor = Parser(options.expressionOneTensor).parseTensorExpression();
    expressionTwoTensors = Parser(options.expressionTwoTensors).parseTensorExpression();
    expressionThreeTensors = Parser(options.expressionThreeTensors).parseTensorExpression();
    expressionFourTensors = Parser(options.expressionFourTensors).parseTensorExpression();

    auto interface = GraphHandle(std::move(rootDimensions), std::vector<const Expand *>{});
    std::unique_ptr<ReductionStage> rootStage;
    std::unique_lock<std::recursive_mutex> lock;
    // Generate Reduce's. This recursively calls ReduceOp::Generate().
    const MutexIndex rootMutexIndex = AbstractStage::GetRootMutexIndex(interface);
    std::tie(rootStage, lock) = ReductionStage::Create(rootMutexIndex, *this, std::move(interface), std::unique_lock<std::recursive_mutex>{});
    this->rootStage = dynamic_cast<ReductionStage *>(stageStore.insert(0, std::move(rootStage), lock));
    lock.unlock();
    auto expander = ThreadPool<ReductionStage *>(numWorkerThreads, [&](ThreadPool<ReductionStage *>& expander, ReductionStage *stage) {
        stage->expand(expander);
    });
    std::size_t numReductionStages = expander.addSync(this->rootStage);
    KAS_DEBUG("Built {} ReductionStage's", numReductionStages);
}

std::size_t Sampler::getExpandAtDepth() {
    std::size_t d = options.depth - 1;
    while (d > 0 && getStats(d).branchingFactor() < 2.0f) {
        --d;
    }
    ++d;
    if (options.depth - d > 5) {
        // This is a bit dangerous.
        d = options.depth - 5;
    } else if (options.depth - d < 3) {
        // It is OK.
        d = options.depth - 3;
    }
    return d;
}

std::string Sampler::statsToString() const {
    std::ostringstream oss;
    for (std::size_t i = 0; const auto& stat: depthwiseStatistics) {
        fmt::format_to(std::ostreambuf_iterator(oss), "Depth {}: {}\n", i, stat.toString());
        ++i;
    }
    return oss.str();
}

const TensorExpression& Sampler::getExpressionForTensorNum(std::size_t num) const {
    switch (num) {
        case 1: return expressionOneTensor;
        case 2: return expressionTwoTensors;
        case 3: return expressionThreeTensors;
        case 4: return expressionFourTensors;
        default: KAS_CRITICAL("Unsupported number of tensors: {}", num);
    }
}

Size Sampler::getFixedDimensionsSize() const {
    auto result = Size::Identity(ctx);
    if (!fixedDimensions.empty()) {
        using FixedDimensionsShapeView = AbstractShape<const std::vector<FixedDimension>&, [](const FixedDimension& fd) -> const Size& { return fd.dim.size(); }>;
        result *= FixedDimensionsShapeView { fixedDimensions }.totalSize();
    }
    return result;
}

Size Sampler::getTotalInputSize() const {
    return inputShape.totalSize() * getFixedDimensionsSize();
}

Size Sampler::getTotalOutputSize() const {
    return outputShape.totalSize() * getFixedDimensionsSize();
}

Size Sampler::getMaxRDomSize() const {
    // We allow at most one matrix multiplication.
    return inputShape.totalSize();
}

int Sampler::remainingChainLength(const Graph& graph, const Dimension& dim) const {
    return static_cast<int>(options.maxChainLength) - graph.colorOf(dim).getHeight();
}

std::optional<Node> Sampler::visit(const std::vector<Next>& path) {
    Node n { this, rootStage };
    for (const auto& next: path) {
        auto nextNode = n.getChild(next);
        if (!nextNode) {
            return std::nullopt;
        }
        n = *nextNode;
    }
    return n;
}

std::optional<std::pair<std::vector<Next>, Node>> Sampler::randomNodeWithPrefix(const std::vector<Next>& prefix) {
    std::vector<Next> path = prefix;
    std::optional<Node> optCur = visit(prefix);
    if (!optCur) {
        return std::nullopt;
    }
    Node cur = *optCur;
    // Recursively visit children.
    while (true) {
        auto children = cur.getChildrenHandles();
        if (children.empty()) {
            // Since the update is performed in multithreaded manner, we cannot assert this.
            // auto stage = cur.tryAsStage();
            // KAS_ASSERT(!stage || stage->getFinalizability() == Finalizability::No);
            break;
        }
        auto next = children[random(children.size())];
        path.emplace_back(next);
        auto nextNode = cur.getChild(next);
        if (!nextNode) {
            return std::nullopt;
        }
        cur = *nextNode;
    };
    return std::optional<std::pair<std::vector<Next>, Node>>(std::in_place, std::move(path), std::move(cur));
}

std::vector<std::pair<std::vector<Next>, Node>> Sampler::randomFinalNodesWithPrefix(const std::vector<Next>& prefix, std::size_t count, std::optional<Next::Type> type, int steps) {
    KAS_ASSERT(steps == 1 || steps == 2, "Only 1-step sampling and 2-step sampling are supported.");

    if (count == 0) return {};

    const std::optional<Node> optCur = visit(prefix);
    if (!optCur) {
        return {};
    }

    const std::size_t expandAtDepth = getExpandAtDepth();
    // + 1 for FinalStage.
    const std::size_t totalDepth = options.depth + 1;

    struct RandomTask {
        bool staged = false;
        std::vector<Next> path;
        Node node;
        std::size_t quota;
        std::optional<Next::Type> type;
    };
    Synchronized<std::vector<RandomTask>> stageArea;
    using Workers = ThreadPool<RandomTask, std::pair<std::vector<Next>, Node>>;
    // Hierarchical sampling.
    auto workers = Workers(numWorkerThreads, [&](Workers& pool, RandomTask task) {
        static thread_local std::mt19937 rng { std::random_device{}() };

        const auto& [staged, path, node, quota, type] = task;
        KAS_ASSERT(quota > 0);

        // If this is final, return it.
        if (node.isFinal()) {
            pool.emplaceResult(std::move(path), std::move(node));
            return;
        }

        // Obtain the children.
        auto children = node.getChildrenHandles();
        if (type != std::nullopt) {
            // Filter by type.
            children.erase(std::remove_if(children.begin(), children.end(), [type = *type](const Next &next) {
                return next.type != type;
            }), children.end());
        }
        if (children.empty()) return;

        // Pick each child with equal probability.
        auto assignment = std::vector<std::size_t>(children.size(), 0);

        if (steps == 1) {
            auto dist = std::uniform_int_distribution<std::size_t>{0, children.size() - 1};
            for (std::size_t i = 0; i < quota; ++i) {
                auto next = dist(rng);
                ++assignment[next];
            }
        } else {
            // Aggregate children by their types.
            std::map<Next::Type, std::vector<std::size_t>> typesAndKeys;
            for (std::size_t i = 0; Next child: children) {
                typesAndKeys[child.type].emplace_back(i++);
            }
            std::vector<std::pair<std::vector<std::size_t>, std::uniform_int_distribution<std::size_t>>> keys;
            keys.reserve(typesAndKeys.size());
            for (auto& [_, key]: typesAndKeys) {
                const std::size_t count = key.size();
                keys.emplace_back(std::move(key), std::uniform_int_distribution<std::size_t>{0, count - 1});
            }
            std::uniform_int_distribution<std::size_t> typeDist{0, keys.size() - 1};
            // Assign quota to each type, then to each key.
            for (std::size_t i = 0; i < quota; ++i) {
                auto& [key, keyDist] = keys[typeDist(rng)];
                auto next = key[keyDist(rng)];
                ++assignment[next];
            }
        }

        // First collect the required nexts and convert them to Nodes.
        std::vector<std::size_t> subQuotas;
        std::vector<Next> subNexts;
        for (std::size_t i = 0; i < children.size(); ++i) {
            std::size_t subQuota = assignment[i];
            if (subQuota == 0) continue;
            subQuotas.emplace_back(subQuota);
            subNexts.emplace_back(children[i]);
        }
        auto optSubNodes = node.getChildren(subNexts);

        KAS_ASSERT(subQuotas.size() == optSubNodes.size() && subQuotas.size() == subNexts.size());

        // Next level of hierarchy.
        std::vector<RandomTask> subTasks;
        for (std::size_t i = 0; i < subQuotas.size(); ++i) {
            auto& optSubNode = optSubNodes[i];
            if (!optSubNode) continue;
            std::size_t subQuota = subQuotas[i];
            std::vector<Next> subPath = path;
            subPath.emplace_back(subNexts[i]);
            subTasks.emplace_back(staged, std::move(subPath), std::move(*optSubNode), subQuota, std::nullopt);
        }

        // For better randomness, to reduce waiting for locks.
        std::shuffle(subTasks.begin(), subTasks.end(), rng);

        // The search tree width gets narrower as we go deeper.
        // So if there are really few steps left, we should just expand, and remove the dead nodes before we proceed.
        const std::size_t nodeDepth = node.depth();
        if (!staged && nodeDepth + 1 >= expandAtDepth) {
            // Expand.
            node.expand(totalDepth - nodeDepth);
            for (auto& task: subTasks) {
                task.staged = true;
            }
            stageArea([&](std::vector<RandomTask>& stageArea) {
                stageArea.reserve(stageArea.size() + subTasks.size());
                std::ranges::move(subTasks, std::back_inserter(stageArea));
            });
        } else {
            // Otherwise just carry on.
            pool.addMultiple(std::move(subTasks));
        }
    });

    workers.addSync(RandomTask{false, prefix, *optCur, count, type});
    getExpander().sync();
    getPruner().sync();
    stageArea([&](std::vector<RandomTask>& stageArea) {
        workers.addMultipleSync(std::move(stageArea));
    });
    auto results = workers.dumpResults();

    std::ranges::sort(results, std::less{}, &std::pair<std::vector<Next>, Node>::second);
    auto [uniqueB, uniqueE] = std::ranges::unique(results, std::equal_to{}, &std::pair<std::vector<Next>, Node>::second);
    results.erase(uniqueB, uniqueE);
    return results;
}

void Sampler::removeFixedDimensions(std::vector<Topmost>& tensors) const {
    auto& inputTensor = tensors.at(0).getDimensions();
    for (const auto& [i, dim]: fixedDimensions | std::views::reverse) {
        // Compare the description here, because it is possible that the dimensions are constructed independently.
        KAS_ASSERT(inputTensor.at(i).description(ctx) == dim.description(ctx));
        inputTensor.erase(inputTensor.begin() + i);
    }
}

void Sampler::sortAllExpansionsAndWeightDimensions(std::vector<Topmost>& tensors) const {
    // First sort the weights in order of hash.
    std::ranges::for_each(tensors | std::views::drop(1), [](Topmost& tensor) {
        tensor.sort();
    });
    // Then sort the expansions in input tensor.
    tensors.at(0).sortExpansions();
    // Then consider if we should disallow permutation of weights.
    if (!options.allowWeightPermutation) {
        auto weights = std::span<Topmost>(tensors.data() + 1, tensors.size() - 1);
        std::ranges::sort(weights, std::less{}, [](const Topmost& topmost) {
            return topmost.getDimensions().at(0).hash();
        });
    }
}

void Sampler::convertTensorsToSearchableForm(std::vector<Topmost>& tensors) const {
    removeFixedDimensions(tensors);
    // TODO!!! No, we should actually look at the rhsOrigin!
    sortAllExpansionsAndWeightDimensions(tensors);
}

std::vector<Topmost> Sampler::convertTensorViewToSearchableTensors(const TensorView& tensorView) const {
    auto tensors = ranges::to<std::vector<Topmost>>(tensorView.getUnderlyingTensors() | std::views::transform(&PureTensor::getContent));
    convertTensorsToSearchableForm(tensors);
    return tensors;
}

Next Sampler::ConvertSearchableTensorsToFinalNext(const std::vector<Topmost>& tensors) {
    return Next { Next::Type::Finalize, NextFinalizeSlot::GetKey(tensors) };
}

std::vector<const PrimitiveOp *> Sampler::ConvertGraphToOps(const Graph& graph) {
    std::vector<const PrimitiveOp *> result;
    // To obtain the path, we need to follow the 3 stages of searching.

    // First, ReductionStage.
    {
        for (const Reduce *op: graph.getReduceIterators()) {
            result.emplace_back(ReduceOp::FromRaw(op));
        }
    }

    // Next, NormalStage.
    {
        std::set<Dimension, Dimension::HashLessThan> completed;
        Graph::AttributeMap<bool> added;
        // Bottom-up.
        auto dfs = [&](const auto& self, const Dimension& dim) -> void {
            if (completed.contains(dim)) return;
            completed.emplace(dim);
            graph.visitAlong(dim, Direction::Down).match(
                [&](const RepeatLikeVertex& v, auto) {
                    bool& addedV = added[v];
                    if (addedV) return;
                    addedV = true;
                    self(self, v[RepeatLikeOp::Branch::Output]);
                    result.emplace_back(&v.op);
                },
                [&](const SplitLikeVertex& v, auto) {
                    bool& addedV = added[v];
                    if (addedV) return;
                    addedV = true;
                    self(self, v[SplitLikeOp::Branch::OutputLhs]);
                    self(self, v[SplitLikeOp::Branch::OutputRhs]);
                    result.emplace_back(&v.op);
                },
                [&](const MergeLikeVertex& v, auto) {
                    bool& addedV = added[v];
                    if (addedV) return;
                    addedV = true;
                    self(self, v[MergeLikeOp::Branch::Output]);
                    result.emplace_back(&v.op);
                },
                [](const ExpandVertex& v, auto) {
                    // Expand is left for later code to handle.
                }
            );
        };
        // We need to generate Next's in canonical order of ShareOp! We need to add ShareOp::IsSharedDimensionCanonical(). TODO.
        for (const Dimension& dim: graph.getTopmost().getAllDimensions()) {
            dfs(dfs, dim);
        }
        // Do not forget to add expansions.
        for (const Expand *exp: graph.getTopmost().getExpansions()) {
            result.emplace_back(dynamic_cast<const ExpandOp *>(exp));
        }
    }

    KAS_ASSERT(result.size() == graph.getOps().size() + graph.getReduceIterators().size());

    return result;
}

std::vector<Next> Sampler::ConvertOpsToNexts(const std::vector<const PrimitiveOp *>& ops) {
    return ranges::to<std::vector<Next>>(ops | std::views::transform(&Next::FromOp<PrimitiveOp>));
}

std::vector<Arc> Sampler::convertOpsToArcs(const std::vector<const PrimitiveOp *>& ops) const {
    return ranges::to<std::vector<Arc>>(ops | std::views::transform([this](const PrimitiveOp *op) {
        return Arc { this, op };
    }));
}

std::vector<Next> Sampler::ConvertSearchableTensorsToPath(const std::vector<Topmost>& tensors) {
    auto handle = GraphHandle::FromInterfaces(tensors);
    auto result = ConvertOpsToNexts(ConvertGraphToOps(handle.buildGraph()));

    // We have now done the previous two stages.
    // Finally, Finalize.
    result.emplace_back(ConvertSearchableTensorsToFinalNext(tensors));

    return result;
}

std::vector<Next> Sampler::convertTensorViewToPath(const TensorView& tensorView) const {
    auto tensors = convertTensorViewToSearchableTensors(tensorView);
    return ConvertSearchableTensorsToPath(tensors);
}

void Sampler::Pruner::handleUpdates(std::set<AbstractStage *>& updates) {
    // Here we just repeatedly try to acquire the lock and update the finalizability.
    // TODO: When the stages release the lock, make them signal the condition variable.
    while (!updates.empty()) {
        std::set<AbstractStage *> remaining;
        for (auto stage: updates) {
            auto lock = stage->tryAcquireLock();
            if (!lock.owns_lock()) {
                remaining.emplace(stage);
            } else {
                stage->updateFinalizability(lock);
            }
        }
        updates.swap(remaining);
    }
}

Sampler::Pruner::Pruner() {
    thread = std::jthread([this](std::stop_token stopToken) {
        std::set<AbstractStage *> processing;
        while (!stopToken.stop_requested()) {
            decltype(inbox) newInbox;
            {
                std::unique_lock lock { mutex };
                cv.wait(lock, stopToken, [this] {
                    return !inbox.empty();
                });
                newInbox.swap(inbox);
            }
            const auto numTasks = newInbox.size();
            while (!newInbox.empty()) {
                processing.emplace(newInbox.front());
                newInbox.pop();
            }
            handleUpdates(processing);
            {
                std::unique_lock lock { mutex };
                completed += numTasks;
                if (clear()) {
                    lock.unlock();
                    cvCompleted.notify_all();
                }
            }
        }
    });
}

void Sampler::Pruner::requestFinalizabilityUpdate(AbstractStage *stage, const AbstractStage *requestor) {
    std::unique_lock lock { mutex };
    inbox.emplace(stage);
    ++committed;
    lock.unlock();
    cv.notify_one();
}

void Sampler::Pruner::sync() {
    std::unique_lock lock { mutex };
    cvCompleted.wait(lock, [this] {
        return clear();
    });
}

Sampler::Pruner::~Pruner() {
    thread.request_stop();
}

void Sampler::Expander::finish() {
    std::unique_lock lock { mutex };
    ++finished;
    if (submitted == finished) {
        ready = true;
        lock.unlock();
        cvReady.notify_all();
    } else {
        lock.unlock();
    }
}

Sampler::Expander::Expander(std::size_t numThreads) {
    for (std::size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back([this](std::stop_token stopToken) {
            static thread_local std::mt19937_64 rng { std::random_device{}() };
            while (!stopToken.stop_requested()) {
                std::unique_lock lock { mutex };
                cv.wait(lock, stopToken, [this] {
                    return !inbox.empty();
                });
                if (!inbox.empty()) {
                    auto [node, layers] = inbox.front();
                    inbox.pop();
                    lock.unlock();
                    if (layers > 0) {
                        auto nexts = node.getChildrenHandles();
                        std::shuffle(nexts.begin(), nexts.end(), rng);
                        auto newNodes = node.getChildren(nexts);
                        for (const auto& newNode: newNodes) {
                            if (newNode.has_value()) {
                                newNode->expand(layers - 1);
                            }
                        }
                    }
                    finish();
                }
            }
        });
    }
}

void Sampler::Expander::expand(Node node, int layers) {
    std::unique_lock lock { mutex };
    ++submitted;
    ready = false;
    inbox.emplace(node, layers);
    lock.unlock();
    cv.notify_one();
}

void Sampler::Expander::expandSync(Node node, int layers) {
    expand(node, layers);
    std::unique_lock lock { mutex };
    cvReady.wait(lock, [this] {
        return ready;
    });
}

void Sampler::Expander::sync() {
    std::unique_lock lock { mutex };
    cvReady.wait(lock, [this] {
        return ready;
    });
}

Sampler::Expander::~Expander() {
    for (auto& thread: threads) {
        thread.request_stop();
    }
}

std::recursive_mutex& Sampler::getMutex(MutexIndex index) {
    return mutexes[index.depth][index.hash % countMutexesInLayer];
}

} // namespace kas
