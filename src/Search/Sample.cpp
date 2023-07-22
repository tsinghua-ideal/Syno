#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/MapReduce.hpp"
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


namespace kas {

void SampleOptions::check() const {
    KAS_ASSERT(dimLowerBound >= 1);
    KAS_ASSERT(dimUpperBound >= dimLowerBound);
    KAS_ASSERT(maximumTensors >= 1);
    KAS_ASSERT(maximumTensors <= 4);
    KAS_ASSERT(maxStridedDimSize > 1);
    KAS_ASSERT(maxUnfoldKernelSize > 1);
    KAS_ASSERT(minimumUnfoldRatio >= 1.0f);
    KAS_ASSERT(minimumMergeRatio >= 1.0f);
    KAS_ASSERT(!requiresOddKernelSizeInUnfold || requiresExactDivision, "requiresOddKernelSizeInUnfold requires requiresExactDivision.");
    KAS_ASSERT(disallowSplitRAboveUnfold + disallowUnfoldLAboveSplit <= 1);
    KAS_ASSERT(disallowMergeWithLargeBlockAboveUnfold + disallowUnfoldLAboveMergeR <= 1);
    KAS_ASSERT(disallowSplitRAboveStride + disallowStrideAboveSplit <= 1);
    KAS_ASSERT(disallowMergeWithLargeBlockAboveStride + disallowStrideAboveMergeR <= 1);
    KAS_ASSERT(disallowUnfoldLAboveShift + disallowShiftAboveUnfold <= 1);
}

std::size_t StageStore::Hash::operator()(const Query& query) const noexcept {
    return query.hash;
}

std::size_t StageStore::Hash::operator()(AbstractStage *stage) const noexcept {
    return stage->getInterface().hash();
}

bool StageStore::Equal::operator()(const Dimensions& lhs, const Dimensions& rhs) const noexcept {
    return std::ranges::equal(lhs, rhs);
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

AbstractStage *StageStore::find(std::size_t depth, const Dimensions& interface, std::unique_lock<std::recursive_mutex>& lock) const {
    KAS_ASSERT(interface.is_sorted(), "Interface is not sorted.");
    KAS_ASSERT(lock.owns_lock());
    Query q = { .interface = interface, .hash = interface.hash() };
    const auto& bucket = bucketsOfStages[depth][q.hash % buckets];
    if (auto it = bucket.find(q); it != bucket.end()) {
        return *it;
    } else {
        return nullptr;
    }
}

AbstractStage *StageStore::insert(std::size_t depth, std::unique_ptr<AbstractStage> stage, std::unique_lock<std::recursive_mutex>& lock) {
    KAS_ASSERT(lock.owns_lock());
    std::size_t bucketIndex = stage->getInterface().hash() % buckets;
    auto& bucket = bucketsOfStages[depth][bucketIndex];
    auto [it, inserted] = bucket.insert(stage.get());
    [[maybe_unused]] auto _ = stage.release();
    return inserted ? *it : nullptr;
}

StageStore::~StageStore() {
    for (auto& bigBucket: bucketsOfStages) {
        for (auto& bucket: bigBucket) {
            for (auto stage: bucket) {
                delete stage;
            }
        }
    }
}

namespace {
    void parseSpecs(const std::vector<std::string>& specs, std::map<std::string, Parser::SizeSpec>& names, const char *prefix) {
        std::size_t unnamed = 0;
        for (const auto& spec: specs) {
            auto result = Parser(spec).parseSizeSpec();
            auto name = result.name();
            if (name) {
                names[*name] = std::move(result);
            } else {
                names[std::string(prefix) + std::to_string(unnamed++)] = std::move(result);
            }
        }
    }
    std::vector<std::pair<std::string, Parser::PureSpec>> contractSpecs(std::map<std::string, Parser::SizeSpec>& specs) {
        std::vector<std::pair<std::string, Parser::PureSpec>> result;
        for (auto&& [name, spec]: specs) {
            result.emplace_back(name, std::move(spec).toPureSpec());
        }
        return result;
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
    options { options },
    stageStore { options.depth + 1, MutexCountFromNumWorkers(numWorkerThreads) },
    countMutexesInLayer { MutexCountFromNumWorkers(numWorkerThreads) },
    mutexes(options.depth + 1),
    pruner {},
    expander { numWorkerThreads }
{
    KAS_ASSERT(numWorkerThreads > 0);
    for (auto& mutexes: this->mutexes) {
        std::vector<std::recursive_mutex> v(countMutexesInLayer);
        mutexes = std::move(v);
    }

    // First parse the variable names in specifications. Unnamed variables are named by x_i and c_i.
    std::map<std::string, Parser::SizeSpec> primaryVars;
    std::map<std::string, Parser::SizeSpec> coefficientVars;
    parseSpecs(primarySpecs, primaryVars, "x_");
    parseSpecs(coefficientSpecs, coefficientVars, "c_");

    // Then we collect the variables in in/out shapes. In case of new variable, add it to primary.
    auto onNewName = [&](const std::string& newName) {
        if (!coefficientVars.contains(newName) && !primaryVars.contains(newName)) {
            // We have to add a default spec for the name.
            primaryVars[newName] = Parser::SizeSpec { .quantity = newName, .maxOccurrences = std::nullopt };
        }
    };
    std::vector<std::string> inputShapeNames = Size::parseNames(inputShape, onNewName);
    std::vector<std::string> outputShapeNames = Size::parseNames(outputShape, onNewName);

    // Put the specs in order.
    auto contractedPrimarySpecs = contractSpecs(primaryVars);
    auto contractedCoefficientSpecs = contractSpecs(coefficientVars);

    // Apply the specs to all variables.
    ctx = BindingContext { contractedPrimarySpecs.size(), contractedCoefficientSpecs.size() };
    ctx.applySpecs(contractedPrimarySpecs, contractedCoefficientSpecs);
    ctx.setMaxVariablesInSize(options.maximumVariablesInSize);
    ctx.setMaxVariablesPowersInSize(options.maximumVariablesPowersInSize);
    ctx.setRequiresExactDivision(options.requiresExactDivision);

    // Parse shape from names.
    this->inputShape = ctx.getShape(inputShapeNames);
    this->outputShape = ctx.getShape(outputShapeNames);

    // Apply the mappings to obtain concrete consts.
    ctx.applyMappings(allMappings);

    this->options.check();
    // Initialize the output iterators.
    for (std::size_t index = 0; const auto& domain: this->outputShape) {
        outputIterators.emplace_back(index++, domain);
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
    }
    for (std::size_t o: boundOutputDimensions | std::views::reverse) {
        this->outputShape.sizes.erase(this->outputShape.sizes.begin() + o);
    }

    // DO NOT modify root after this, because Dimension references by address these iterators.
    for (const auto& it: outputIterators) {
        // Exclude the bound dimensions.
        if (std::ranges::binary_search(boundOutputDimensions, it.getIndex())) {
            continue;
        }
        root.emplace_back(&it);
    }

    expressionOneTensor = Parser(options.expressionOneTensor).parseTensorExpression();
    expressionTwoTensors = Parser(options.expressionTwoTensors).parseTensorExpression();
    expressionThreeTensors = Parser(options.expressionThreeTensors).parseTensorExpression();
    expressionFourTensors = Parser(options.expressionFourTensors).parseTensorExpression();

    Dimensions interface = getRootInterface();
    interface.sort();
    std::unique_ptr<ReductionStage> rootStage;
    std::unique_lock<std::recursive_mutex> lock;
    // Generate MapReduce's. This recursively calls MapReduceOp::Generate().
    std::tie(rootStage, lock) = ReductionStage::Create(*this, std::move(interface), std::unique_lock<std::recursive_mutex>{});
    this->rootStage = dynamic_cast<ReductionStage *>(stageStore.insert(0, std::move(rootStage), lock));
    ReductionStage::Expander expander { numWorkerThreads };
    expander.addRoot(this->rootStage, std::move(lock));
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

Size Sampler::getTotalOutputSize() const {
    Size result = outputShape.totalSize();
    if (!fixedDimensions.empty()) {
        using FixedDimensionsShapeView = AbstractShape<const std::vector<FixedDimension>&, [](const FixedDimension& fd) -> const Size& { return fd.dim.size(); }>;
        result = result * FixedDimensionsShapeView { fixedDimensions }.totalSize();
    }
    return result;
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
        auto cnt = cur.countChildren();
        if (cnt == 0) {
            // Since the update is performed in multithreaded manner, we cannot assert this.
            // auto stage = cur.tryAsStage();
            // KAS_ASSERT(!stage || stage->getFinalizability() == Finalizability::No);
            break;
        }
        auto next = cur.getChildrenHandles()[random(cnt)];
        path.emplace_back(next);
        auto nextNode = cur.getChild(next);
        if (!nextNode) {
            return std::nullopt;
        }
        cur = *nextNode;
    };
    return std::optional<std::pair<std::vector<Next>, Node>>(std::in_place, std::move(path), std::move(cur));
}

void Sampler::ConvertTensorViewToSearchableOrder(std::vector<std::vector<Dimension>>& tensorView) {
    // First sort the weights in order of hash. This somewhat duplicates the functionality in Forward::buildTensorView(). TODO
    std::ranges::for_each(tensorView | std::views::drop(1), [](std::vector<Dimension>& dims) {
        std::ranges::sort(dims, Dimension::HashLessThan{});
    });
}

std::vector<Next> Sampler::ConvertGraphToPath(const Graph& graph) {
    std::vector<Next> result;
    // To obtain the path, we need to follow the 3 stages of searching.

    // First, ReductionStage.
    {
        for (const MapReduce *op: graph.getMapReduceIterators()) {
            result.emplace_back(Next::Type::MapReduce, op->hash());
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
                    result.emplace_back(Next::TypeOf(v.op.getType()), v.op.opHash());
                },
                [&](const SplitLikeVertex& v, auto) {
                    bool& addedV = added[v];
                    if (addedV) return;
                    addedV = true;
                    self(self, v[SplitLikeOp::Branch::OutputLhs]);
                    self(self, v[SplitLikeOp::Branch::OutputRhs]);
                    result.emplace_back(Next::TypeOf(v.op.getType()), v.op.opHash());
                },
                [&](const MergeLikeVertex& v, auto) {
                    bool& addedV = added[v];
                    if (addedV) return;
                    addedV = true;
                    self(self, v[MergeLikeOp::Branch::Output]);
                    result.emplace_back(Next::TypeOf(v.op.getType()), v.op.opHash());
                }
            );
        };
        // We need to generate Next's in canonical order of ShareOp! We need to add ShareOp::IsSharedDimensionCanonical(). TODO.
        for (const Dimension& dim: graph.getTopmost()) {
            dfs(dfs, dim);
        }
    }

    return result;
}

std::vector<Next> Sampler::convertTensorsToPath(const std::vector<std::vector<Dimension>>& tensors) const {
    Graph::Builder builder;
    builder.addTopmost(tensors | std::views::join);
    Graph graph = builder.build();

    auto result = ConvertGraphToPath(graph);

    // We have now done the previous two stages.
    // Finally, Finalize.
    {
        // The fixed dimensions should be removed first.
        std::vector<std::vector<Dimension>> tensorsInSearchTree;
        std::ranges::copy(tensors, std::back_inserter(tensorsInSearchTree));
        ConvertTensorViewToSearchableOrder(tensorsInSearchTree);
        auto& inputTensor = tensorsInSearchTree.at(0);
        for (const auto& [i, _]: fixedDimensions | std::views::reverse) {
            inputTensor.erase(inputTensor.begin() + i);
        }
        result.emplace_back(Next::Type::Finalize, NextFinalizeSlot::GetKey(tensorsInSearchTree));
    }

    return result;
}

std::optional<std::vector<Arc>> Sampler::convertPathToArcs(const std::vector<Next>& path) {
    std::vector<Arc> result;
    Node n { this, rootStage };
    for (const auto& next: path) {
        auto nextArc = n.getArcFromHandle(next);
        if (!nextArc) {
            return std::nullopt;
        }
        result.emplace_back(*nextArc);
        n = n.getChildFromArc(*nextArc);
    }
    return std::optional<std::vector<Arc>>(std::in_place, std::move(result));
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
            while (!newInbox.empty()) {
                processing.emplace(newInbox.front());
                newInbox.pop();
            }
            handleUpdates(processing);
        }
    });
}

void Sampler::Pruner::requestFinalizabilityUpdate(AbstractStage *stage, const AbstractStage *requestor) {
    std::unique_lock lock { mutex };
    inbox.emplace(stage);
    lock.unlock();
    cv.notify_one();
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
                        for (auto next: nexts) {
                            auto newNode = node.getChild(next);
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

}

Sampler::Expander::~Expander() {
    for (auto& thread: threads) {
        thread.request_stop();
    }
}

std::recursive_mutex& Sampler::getMutex(std::size_t depth, const Dimensions& interface) {
    return mutexes[depth][interface.hash() % countMutexesInLayer];
}

} // namespace kas
