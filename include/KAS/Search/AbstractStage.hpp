#pragma once

#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Sample.hpp"


namespace kas {

// Finalizability of a Stage.
enum class Finalizability {
    Maybe, // Default state.
    Yes, // Determined by expanding subtrees, and see if there are final nodes.
    No, // Determined by expanding all subtrees or conservative experiments.
};
inline Finalizability operator+(const Finalizability& a, const Finalizability& b) {
    // Observe "+" for states is commutative and associative, where the identity is Finalizability::No.
    bool foundYes = a == Finalizability::Yes || b == Finalizability::Yes;
    bool allNo = a == Finalizability::No && b == Finalizability::No;
    if (foundYes) {
        return Finalizability::Yes;
    } else if (allNo) {
        return Finalizability::No;
    } else {
        return Finalizability::Maybe;
    }
}
inline Finalizability& operator+=(Finalizability& a, const Finalizability& b) {
    a = a + b;
    return a;
}

struct MutexIndex {
    // layerIndex
    std::size_t depth;
    // hash % mutexCount == mutexIndex
    std::size_t hash;
};

// A thread-safe data structure for Nodes. Only immutable members are allowed. Mutable members are stored in AbstractStageBase.
class AbstractStage {
    template<typename DerivedStageType>
    friend class AbstractStageBase;
    friend class Sampler::Pruner;

public:
    using Mutex = std::recursive_mutex;
    using Lock = std::unique_lock<Mutex>;
protected:
    [[nodiscard]] Lock acquireLock() const { return Lock { mutex }; }
    [[nodiscard]] Lock tryAcquireLock() const { return Lock { mutex, std::try_to_lock }; }
private:
    Lock initialLock;
    Mutex& mutex;
    Lock obtainInitialLock();

    // Parents of this Stage.
    std::vector<AbstractStage *> parents;

    Finalizability state = Finalizability::Maybe;
    bool isFinalizabilityDetermined() const { return state != Finalizability::Maybe; }

    std::atomic<int> expandedLayers = 0;
    bool tryExpandingMoreLayers(int layers);

protected:
    Sampler& sampler;

    // The interface decides the hash. Other properties are computed.
    GraphHandle interface;

    // Stages with identical interfaces must be of the same depth.
    std::size_t depth;

    // A statistics counter for all Ops.
    Next::OpTypeCounter existingOps {};

    // When the finalizability is determined, call parents to update their finalizability.
    // When in construction, this function does not call the update. Otherwise, update is called.
    void determineFinalizability(Finalizability yesOrNo, bool propagate);

    // Node pointers. We are searching bottom-up, so the children are actually closer to the input tensor.
    NextSlotStore nextSlotStore;

    void requestFinalizabilityUpdate(const AbstractStage *requestor);

public:
    KAS_STATISTICS_DEF(
        Creations,
        ChildrenReduce,
        ChildrenExpand,
        ChildrenShift,
        ChildrenStride,
        ChildrenSplit,
        ChildrenUnfold,
        ChildrenMerge,
        ChildrenShare,
        FinalizabilityMaybe,
        FinalizabilityYes,
        FinalizabilityNo,
    )
    // Create a root stage.
    AbstractStage(Sampler &sampler, GraphHandle interface, Lock lock);
    // Create a non-root stage.
    AbstractStage(GraphHandle interface, AbstractStage& creator, std::optional<Next::Type> optionalDeltaOp, Lock lock);
    Lock addParent(AbstractStage &parent);
    void addParent(AbstractStage &parent, Lock &lock);
    AbstractStage *arbitraryParent() const;

    // Disallow copy or move.
    AbstractStage(const AbstractStage&) = delete;
    AbstractStage(AbstractStage&&) = delete;

    const GraphHandle& getInterface() const { return interface; }
    std::size_t getDepth() const { return depth; }

    // Compute from Sampler.
    std::size_t remainingDepth() const;
    DepthwiseStatistics& getStats() const;

    template<PrimitiveOpImpl Op>
    int existingOp() const { return existingOps[Next::TypeOf<Op>()]; }

    // The state.
    inline Finalizability getFinalizability() const {
        if (isFinalizabilityDetermined()) {
            // Fast path. Here we do not need to acquire the lock.
            // This is because once the state is determined, it will never change.
            // You may ask: hey, what if we read a value of a stage that has not been fully initialized yet, in which case `state` may be an arbitrary value?
            // Think about it. How did you obtain this stage at all? You must have synchronized with a predecessor of this stage, which created this stage.
            // So this fast path is thread-safe.
            return state;
        }
        Lock lock = acquireLock();
        return state;
    }
    Finalizability getFinalizability(Lock& lock) const {
        KAS_ASSERT(lock.owns_lock());
        return state;
    }

    // This must be called by Pruner!
    virtual void updateFinalizability(Lock& lock) = 0;

    virtual bool possibleToFinalizeByExperimenting() const { return true; }
    void recomputeShapeDistance() const;
    virtual ShapeDistance getShapeDistance() const = 0;

    virtual Node toNode() = 0;

    // Python.
    std::size_t hash() const;
    virtual std::size_t countChildren() = 0;
    virtual std::vector<Next> getChildrenHandles() = 0;
    virtual std::vector<Arc> getChildrenArcs() = 0;
    virtual std::optional<Arc> getArcFromHandle(Next next) = 0;
    virtual std::optional<Node> getChild(Next next) = 0;
    virtual std::vector<std::optional<Node>> getChildren(const std::vector<Next>& nexts) = 0;
    virtual std::vector<std::optional<Node>> getChildren(const std::vector<Arc>& arcs) = 0;
    virtual bool canAcceptArc(Arc arc) = 0;
    virtual std::optional<Node> getChild(Arc arc) = 0;
    std::string description() const;

    inline MutexIndex getMutexIndex() const {
        return { .depth = depth, .hash = interface.hash() };
    }
    inline MutexIndex getNextMutexIndex(bool hasOp, const GraphHandle& interface) const {
        return { .depth = depth + static_cast<std::size_t>(hasOp), .hash = interface.hash() };
    }
    static inline MutexIndex GetRootMutexIndex(const GraphHandle& interface) {
        return { .depth = 0, .hash = interface.hash() };
    }

    void expand(int layers);
    void expandSync(int layers);

    virtual ~AbstractStage() = default;
};

// Requirements of a non-thread-safe stage implementation.
template<typename DerivedStageType>
concept StageImpl = requires(DerivedStageType child, const DerivedStageType::CollectedFinalizabilities& collected) {
    { child.collectFinalizabilities() } -> std::convertible_to<typename DerivedStageType::CollectedFinalizabilities>;
    { child.removeDeadChildrenFromSlots(collected) };
    { child.removeAllChildrenFromSlots() };
    { child.checkForFinalizableChildren(collected) } -> std::convertible_to<Finalizability>;

    // The standard functions required by AbstractStage.
    // They are the non-thread-safe versions. This class is a thread-safe wrapper for them.
    { child.countChildrenImpl() } -> std::convertible_to<std::size_t>;
    { child.getChildrenHandlesImpl() } -> std::convertible_to<std::vector<Next>>;
    { child.getChildrenArcsImpl() } -> std::convertible_to<std::vector<Arc>>;
    { child.getArcFromHandleImpl(std::declval<Next>()) } -> std::convertible_to<std::optional<Arc>>;
    { child.getChildImpl(std::declval<Next>()) } -> std::convertible_to<std::optional<Node>>;
    { child.canAcceptArcImpl(std::declval<Arc>()) } -> std::convertible_to<bool>;
    { child.getChildImpl(std::declval<Arc>()) } -> std::convertible_to<std::optional<Node>>;
};

// A CRTP template class for AbstractStage. All subclasses must inherit from this class.
// This class is associated with a mutex in Sampler. Each function first acquires the lock, and then dispatch the functionality to the subclass.
template<typename DerivedStageType>
class AbstractStageBase: public AbstractStage {
    DerivedStageType& derived() { return static_cast<DerivedStageType&>(*this); }
    const DerivedStageType& derived() const { return static_cast<const DerivedStageType&>(*this); }
protected:
    using Base = AbstractStageBase<DerivedStageType>;

    template<typename Slot, std::ranges::input_range R, typename F>
    requires std::convertible_to<std::invoke_result_t<F, std::ranges::range_reference_t<R>>, Slot>
    void fillSlots(GenericNextSlotStore<Slot>& slotStore, R&& rawStream, F&& builder, bool isFinal = false) {
        std::size_t filled = slotStore.fill(std::forward<R>(rawStream), std::forward<F>(builder));
        if (isFinal) {
            getStats().addFinalChildren(filled);
        }
        else {
            getStats().addNonFinalChildren(filled);
        }
    }

    using CollectedFinalizabilities = std::vector<Finalizability>;

    CollectedFinalizabilities collectFinalizabilities() {
        return nextSlotStore.map([&](const NextStageSlot& slot) {
            return slot.nextStage->getFinalizability();
        });
    }

    void removeDeadChildrenFromSlots(const CollectedFinalizabilities& collected) {
        std::size_t removed = nextSlotStore.removeByIndex([&](std::size_t index) {
            return collected[index] == Finalizability::No;
        });
        getStats().removeNonFinalChildren(removed);
    }

    void removeAllChildrenFromSlots() {
        std::size_t removed = nextSlotStore.clear();
        getStats().removeNonFinalChildren(removed);
    }

    Finalizability checkForFinalizableChildren(const CollectedFinalizabilities& collected) const {
        // Check children. Yes if any Yes, No if all No.
        Finalizability fin = Finalizability::No;
        for (Finalizability f: collected) {
            fin += f;
        }
        return fin;
    }

private:
    // This function is only executed by a background update thread (Pruner).
    // Update finalizability.
    // If the state is updated, then notify the parents by enqueing them in Pruner.
    void updateFinalizability(Lock& lock) final override {
        KAS_ASSERT(lock.owns_lock());

        auto collected = derived().collectFinalizabilities();

        switch (state) {
        case Finalizability::Maybe: {
            // First carry out check.
            Finalizability newFinalizability = derived().checkForFinalizableChildren(collected);
            if (newFinalizability == Finalizability::Maybe) {
                // Well, nothing new.
                // But we still need to remove the dead ends.
                derived().removeDeadChildrenFromSlots(collected);
            } else {
                // Otherwise, we have determined the finalizability.
                // Do the update first.
                switch (newFinalizability) {
                case Finalizability::Yes:
                    derived().removeDeadChildrenFromSlots(collected);
                    break;
                case Finalizability::No:
                    derived().removeAllChildrenFromSlots();
                    break;
                default: KAS_UNREACHABLE();
                }
                // Then determine.
                determineFinalizability(newFinalizability, true);
                // This call puts this stage in Pruner's queue.
            }
            break;
        }
        case Finalizability::Yes: {
            // Still need to remove dead ends.
            derived().removeDeadChildrenFromSlots(collected);
            break;
        }
        case Finalizability::No: {
            // Usually this is not needed, becase when we determined Finalizability::No, we would have removed all children.
            derived().removeAllChildrenFromSlots();
            break;
        }
        default: KAS_UNREACHABLE();
        }
    }

protected:
    // Apply the Op to obtain the next Stage.
    template<typename ChildStageType = DerivedStageType>
    std::pair<ChildStageType *, Lock> getNextOp(const PrimitiveOp *op) {
        // When this gets called, we are holding the lock of this stage.
        StageStore& store = sampler.getStageStore();
        const bool nextLayer = op != nullptr;
        GraphHandle newInterface = nextLayer ? op->applyToInterface(interface) : interface;
        const MutexIndex mutexIndex = getNextMutexIndex(nextLayer, newInterface);
        Lock lock = std::unique_lock { sampler.getMutex(mutexIndex) };
        if (AbstractStage *found = store.find(depth + nextLayer, newInterface, lock); found) {
            found->addParent(*this, lock);
            auto childStage = dynamic_cast<ChildStageType *>(found);
            KAS_ASSERT(childStage);
            return { childStage, std::move(lock) };
        } else {
            auto [tempStage, newLock] = ChildStageType::Create(
                mutexIndex,
                std::move(newInterface),
                *this,
                nextLayer ? std::make_optional<Next::Type>(Next::TypeOf(op->getType())) : std::nullopt,
                std::move(lock)
            );
            // Perform experimental finalization, i.e., compute the ShapeComplexity of the interface.
            if (!tempStage->possibleToFinalizeByExperimenting()) {
                // If proved to be not finalizable, no need to store this.
                return { nullptr, Lock() };
            } else {
                if (auto it = store.insert(depth + nextLayer, std::move(tempStage), newLock); it) {
                    auto childStage = dynamic_cast<ChildStageType *>(it);
                    KAS_ASSERT(childStage);
                    return { childStage, std::move(newLock) };
                } else {
                    KAS_CRITICAL("StageStore::insert() failed.");
                }
            }
        }
    }
    template<typename ChildStageType = DerivedStageType>
    ChildStageType *getNextOpWithoutLock(const PrimitiveOp *op) {
        auto [childStage, lock] = getNextOp<ChildStageType>(op);
        return childStage;
    }

    ShapeDistance getShapeDistanceImpl() const {
        return { remainingDepth(), 0 };
    }
    std::size_t countChildrenImpl() const {
        return nextSlotStore.size();
    }
    std::vector<Next> getChildrenHandlesImpl() const {
        return nextSlotStore.toNexts();
    }
    std::vector<Arc> getChildrenArcsImpl() const {
        return nextSlotStore.toArcs(&sampler);
    }
    std::optional<Arc> getArcFromHandleImpl(Next next) const {
        return nextSlotStore.findTransform<Arc>(next, [this](const NextStageSlot& slot) -> Arc {
            return slot.toArc(&sampler);
        });
    }
    std::optional<Node> getChildImpl(Next next) {
        return nextSlotStore.findTransform<Node>(next, [this](const NextStageSlot& slot) -> Node {
            auto childStage = dynamic_cast<DerivedStageType *>(slot.nextStage);
            KAS_ASSERT(childStage);
            return { &sampler, childStage };
        });
    }
    bool canAcceptArcImpl(Arc arc) {
        return arc.match<bool>(
            [&](const PrimitiveOp *op) -> bool {
                return op->canApplyToInterface(interface);
            },
            [&](const FinalizeOp *op) -> bool {
                return op->toGraphHandle() == interface;
            }
        );
    }

    AbstractStageBase(Sampler& sampler, GraphHandle interface, Lock lock):
        AbstractStage(sampler, std::move(interface), std::move(lock))
    {
        static_assert(StageImpl<DerivedStageType>);
    }
    AbstractStageBase(GraphHandle interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock):
        AbstractStage(std::move(interface), creator, std::move(deltaOp), std::move(lock))
    {
        static_assert(StageImpl<DerivedStageType>);
    }

public:
    // The initial lock must be obtained. So we make the constructors invisible to outside.
    template<typename... Args>
    [[nodiscard]] static std::pair<std::unique_ptr<DerivedStageType>, Lock> Create(const MutexIndex& mutexIndex, Args&&... args) {
        auto stage = std::make_unique<DerivedStageType>(std::forward<Args>(args)...);
        Lock lock = stage->obtainInitialLock();
        return { std::move(stage), std::move(lock) };
    }

    ShapeDistance getShapeDistance() const final override {
        Lock lock = acquireLock();
        return derived().getShapeDistanceImpl();
    }

    Node toNode() final override {
        return Node { &sampler, &derived() };
    }

    std::size_t countChildren() final override {
        Lock lock = acquireLock();
        return derived().countChildrenImpl();
    }
    std::vector<Next> getChildrenHandles() final override {
        Lock lock = acquireLock();
        return derived().getChildrenHandlesImpl();
    }
    std::vector<Arc> getChildrenArcs() final override {
        Lock lock = acquireLock();
        return derived().getChildrenArcsImpl();
    }
    std::optional<Arc> getArcFromHandle(Next next) final override {
        Lock lock = acquireLock();
        return derived().getArcFromHandleImpl(next);
    }
    std::optional<Node> getChild(Next next) final override {
        Lock lock = acquireLock();
        return derived().getChildImpl(next);
    }
    std::vector<std::optional<Node>> getChildren(const std::vector<Next>& nexts) final override {
        Lock lock = acquireLock();
        return ranges::to<std::vector<std::optional<Node>>>(
            nexts
            | std::views::transform([&](Next next) -> std::optional<Node> {
                return derived().getChildImpl(next);
            })
        );
    }
    std::vector<std::optional<Node>> getChildren(const std::vector<Arc>& arcs) final override {
        Lock lock = acquireLock();
        return ranges::to<std::vector<std::optional<Node>>>(
            arcs
            | std::views::transform([&](Arc arc) -> std::optional<Node> {
                return derived().getChildImpl(arc);
            })
        );
    }
    bool canAcceptArc(Arc arc) final override {
        Lock lock = acquireLock();
        return derived().canAcceptArcImpl(arc);
    }
    std::optional<Node> getChild(Arc arc) final override {
        Lock lock = acquireLock();
        return derived().getChildImpl(arc);
    }
};

} // namespace kas
