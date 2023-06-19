#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <variant>
#include <vector>

#include <gtest/gtest_prod.h>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/AbstractStage.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Transforms/PrimitiveOpStore.hpp"
#include "KAS/Utils/Hash.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class Sampler;
class NormalStage;

class NormalStageStore {
public:
    struct Hash {
        using is_transparent = void;
        std::size_t operator()(const ColoredInterface& interface) const noexcept;
        std::size_t operator()(const NormalStage * nStage) const noexcept;
    };
    struct Equal {
        using is_transparent = void;
        bool operator()(const ColoredInterface& lhs, const ColoredInterface& rhs) const noexcept;
        bool operator()(const ColoredInterface& lhs, const NormalStage *rhs) const noexcept;
        bool operator()(const NormalStage *lhs, const ColoredInterface& rhs) const noexcept;
        bool operator()(const NormalStage *lhs, const NormalStage *rhs) const noexcept;
    };

private:
    PrimitiveOpStore opStore;
    std::unordered_set<NormalStage *, Hash, Equal> interfaces;

public:
    PrimitiveOpStore& getOpStore() { return opStore; }
    NormalStage *find(const ColoredInterface& interface) const;
    bool insert(NormalStage *nStage);
    ~NormalStageStore();
};

class NormalStage final: public AbstractStage {
    // The interface decides the hash. Other properties are computed.
    ColoredInterface interface;

    // Lazily generate children.
    bool childrenGenerated = false;
    // Node pointers. The nodes are lazily computed. We are searching bottom-up, so the children are actually closer to the input.
    NextSlotStore<NextFinalizeSlot> nextFinalizations;
    NextOpStores<MapReduceOp, ShiftOp, StrideOp, SplitOp, UnfoldOp, MergeOp, ShareOp> nextOpStores;

    NormalStageStore& getNormalStageStore();

    void removeDeadChildrenFromSlots() override;
    void removeAllChildrenFromSlots() override;
    Finalizability checkForFinalizableChildren() const override;

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guardGeneratedChildren();

    // Execute the finalization to obtain TensorView.
    std::shared_ptr<TensorView> getFinalize(std::size_t key) const;

    // Apply the Op to obtain NormalStage.
    template<PrimitiveOpImpl Op>
    NormalStage *getNextOp(const Op *op) {
        NormalStageStore& store = getNormalStageStore();
        auto newInterface = op->applyToInterface(interface);
        if (NormalStage *found = store.find(newInterface); found) {
            found->addParent(*this);
            return found;
        } else {
            auto tempStage = std::make_unique<NormalStage>(std::move(newInterface), *this, Next::TypeOf<Op>());
            if(store.insert(tempStage.get())) {
                return tempStage.release();
            } else {
                KAS_CRITICAL("NormalStageStore::insert() failed.");
            }
        }
    }

    // This is for pruning. We experimentally finalize this stage, and conservatively exclude the stage if it is not possible to finalize.
    bool possibleToFinalizeByExperimenting() const;

    std::size_t uncheckedCountChildren() const;
    std::vector<Next> uncheckedGetChildrenHandles() const;
    const NextFinalizeSlot *getChildFinalizeSlot(std::size_t key) const;
    template<PrimitiveOpImpl Op>
    const NextOpSlot<Op> *getChildSlot(std::size_t key) const {
        return nextOpStores.get<Op>().getSlot(key);
    }
    std::optional<Node> uncheckedGetChild(Next next) const;
    std::optional<std::string> uncheckedGetChildDescription(Next next);

    template<typename F>
    inline auto guarded(F&& f) -> decltype(f()) {
        guardGeneratedChildren();
        return AbstractStage::guarded(std::forward<F>(f));
    }

public:
    KAS_STATISTICS_DEF(
        ChildrenFinalize,
        FinalizabilityCheckInvocations,
        TooManyWeights,
        ShapeDeviatesTooMuch,
    );

    // The root.
    NormalStage(Sampler& sampler);
    NormalStage(ColoredInterface&& interface, AbstractStage& creator, std::optional<Next::Type> deltaOp);

    const ColoredInterface& getInterface() const { return interface; }

    std::size_t hash() const override { return NormalStageStore::Hash{}(interface); }
    std::size_t countChildren() override;
    std::vector<Next> getChildrenHandles() override;
    std::optional<Node> getChild(Next next) override;
    std::optional<std::string> getChildDescription(Next next) override;
    std::string description() const override;
};

} // namespace kas
