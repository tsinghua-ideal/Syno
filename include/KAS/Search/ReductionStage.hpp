#pragma once

#include <limits>
#include <variant>

#include "KAS/Core/MapReduce.hpp"
#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Search/NormalStage.hpp"


namespace kas {

class ReductionStage final: public AbstractStage {
public:
    struct NextReductionSlot: NextSlot<Next::Type::MapReduce> {
        std::unique_ptr<ReductionStage> next;
        static std::size_t GetKey(const MapReduceOp *op) { return op->hash(); }
    };

private:
    std::vector<const MapReduceOp *> reductions;

    NextSlotStore<NextReductionSlot> nextReductions;

    // The stage that is directly constructed, without appending any reduction.
    std::unique_ptr<NormalStage> nStage;

    void expand();

    void removeDeadChildrenFromSlots() override;
    void removeAllChildrenFromSlots() override;
    Finalizability checkForFinalizableChildren() const override;

    std::size_t uncheckedCountChildren();
    std::vector<Next> uncheckedGetChildrenHandles();
    const NextReductionSlot& getChildSlot(std::size_t key);
    Node uncheckedGetChild(Next next);
    std::string uncheckedGetChildDescription(Next next);

public:
    ReductionStage(ReductionStage& current, const MapReduceOp *nextReduction);
    // This is the root.
    ReductionStage(Sampler& sampler);

    std::size_t hash() const;

    const MapReduceOp *lastReduction() const { return reductions.size() ? reductions.back() : nullptr; }
    Interface toInterface() const;

    std::size_t countChildren();
    std::vector<Next> getChildrenHandles();
    Node getChild(Next next);
    std::string getChildDescription(Next next);
    std::string description() const;
};

} // namespace kas
