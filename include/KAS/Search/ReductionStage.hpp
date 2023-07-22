#pragma once

#include <condition_variable>

#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Transforms/MapReduce.hpp"


namespace kas {

class ReductionStage final: public AbstractStageBase<ReductionStage> {
public:
    class Expander {
        std::mutex mutex;
        std::condition_variable_any cv;
        std::condition_variable cvReady;
        std::queue<ReductionStage *> queue;
        std::vector<std::jthread> workers;
        std::size_t submitted = 0;
        std::size_t completed = 0;
    public:
        Expander(std::size_t numWorkers);
        Expander(const Expander&) = delete;
        Expander(Expander&&) = delete;
        // Async.
        void add(ReductionStage *stage, Lock lock);
        // Synchronize.
        void addRoot(ReductionStage *stage, Lock lock);
        ~Expander();
    };

private:
    friend class AbstractStageBase<ReductionStage>;

    // The stage that is directly constructed, without appending any reduction.
    std::unique_ptr<NormalStage> nStage;

    bool expanded = false;
    void expand(Expander& expander);

    struct CollectedFinalizabilities: Base::CollectedFinalizabilities {
        Finalizability nStageFinalizability;
    };
    CollectedFinalizabilities collectFinalizabilities();
    Finalizability checkForFinalizableChildren(const CollectedFinalizabilities& collected) const;

public:
    // This is the root.
    ReductionStage(Sampler& sampler, Dimensions interface, Lock lock);
    ReductionStage(Dimensions interface, AbstractStage& creator, std::optional<Next::Type> deltaOp, Lock lock);

    std::size_t countChildrenImpl() const;
    std::vector<Next> getChildrenHandlesImpl();
    std::vector<Arc> getChildrenArcsImpl();
    std::optional<Arc> getArcFromHandleImpl(Next next);
    std::optional<Node> getChildImpl(Next next);
    bool canAcceptArcImpl(Arc arc);
    Node getChildImpl(Arc arc);
};

} // namespace kas
