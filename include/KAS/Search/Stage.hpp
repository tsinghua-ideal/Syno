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
#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Node.hpp"
#include "KAS/Transforms/DimensionStore.hpp"
#include "KAS/Utils/Hash.hpp"
#include "KAS/Utils/Statistics.hpp"


namespace kas {

class Sampler;
class Stage;

class StageStore {
    DimensionStore dimensionStore;
    struct Hash {
        using is_transparent = void;
        std::size_t operator()(const ColoredInterface& interface) const noexcept;
        std::size_t operator()(const Stage * stage) const noexcept;
    };
    struct Equal {
        using is_transparent = void;
        bool operator()(const ColoredInterface& lhs, const ColoredInterface& rhs) const noexcept;
        bool operator()(const ColoredInterface& lhs, const Stage *rhs) const noexcept;
        bool operator()(const Stage *lhs, const ColoredInterface& rhs) const noexcept;
        bool operator()(const Stage *lhs, const Stage *rhs) const noexcept;
    };
    std::unordered_set<Stage *, Hash, Equal> interfaces;

public:
    DimensionStore& dimStore() { return dimensionStore; }
    Stage *find(const ColoredInterface& interface) const;
    bool insert(Stage *stage);
    ~StageStore();
};

class Stage {
public:
    struct NextFinalizeSlot: NextSlot<Next::Type::Finalize> {
        FinalizeOp finalization;
        template<TensorRange TR>
        static std::size_t GetKey(TR&& tensors) { return std::hash<std::vector<Interface>>{}(tensors); }
    };

    template<typename Op>
    struct NextOpSlot: NextSlot<Next::TypeOf<Op>()> {
        const Op *op;
        Stage *nextStage;
        static std::size_t GetKey(const Op *op) { return op->opHash(); }
    };

    template<typename Op>
    using NextOpStore = NextSlotStore<NextOpSlot<Op>>;
    template<typename... Ops>
    struct NextOpStores {
        std::tuple<NextOpStore<Ops>...> stores;
        template<typename Op>
        NextOpStore<Op>& get() {
            return std::get<NextOpStore<Op>>(stores);
        }
        template<typename Op>
        const NextOpStore<Op>& get() const {
            return std::get<NextOpStore<Op>>(stores);
        }
    };

private:
    // The interface decides the hash. Other properties are computed.
    ColoredInterface interface;

    // Lazily generate children.
    bool childrenGenerated = false;
    // Children handles.
    std::vector<Next> nexts;
    // Node pointers. The nodes are lazily computed. We are searching bottom-up, so the children are actually closer to the input.
    NextSlotStore<NextFinalizeSlot> nextFinalizations;
    NextOpStores<ShiftOp, StrideOp, SplitOp, UnfoldOp, MergeOp, ShareOp> nextOpStores;

    // Metadata.
    Sampler& sampler;
    StageStore& getStageStore();
    // std::vector<std::reference_wrapper<const Size>> missingSizes;
    std::size_t depth; // Stages with identical interfaces must be of the same depth.

    std::size_t remainingDepth() const;

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guard();

    // Execute the finalization to obtain TensorView.
    std::shared_ptr<TensorView> getFinalize(std::size_t key);

    // Apply the Op to obtain Stage.
    template<typename Op>
    Stage *getNextOp(const Op *op) {
        StageStore& store = getStageStore();
        auto newInterface = op->applyToInterface(interface);
        if (Stage *found = store.find(newInterface); found) {
            return found;
        } else {
            auto tempStage = std::make_unique<Stage>(std::move(newInterface), sampler, depth + 1);
            if(store.insert(tempStage.get())) {
                return tempStage.release();
            } else {
                KAS_CRITICAL("StageStore::insert() failed.");
            }
        }
    }

    // Remove the Op from store, deleting its enclosing dimensions as well.
    void removeOp(const PrimitiveOp *op) {
        // TODO
    }

public:
    Stage(auto&& interface, Sampler& sampler, std::size_t depth):
        interface { std::forward<decltype(interface)>(interface) },
        sampler { std::forward<decltype(sampler)>(sampler) },
        depth { depth }
    {
        // Compute missing sizes. TODO.
    }

    KAS_STATISTICS_DEF(
        Creations,
        ChildrenFinalize,
        ChildrenShift,
        ChildrenStride,
        ChildrenSplit,
        ChildrenUnfold,
        ChildrenMerge,
        ChildrenShare,
        FinalizabilityCheckInvocations,
        TooManyStridedDims,
        TooManyWeights,
        TooFewElementsInInputTensor,
        ShapeDeviatesTooMuch,
    );
    // This is for pruning. We experimentally finalize this stage, and conservatively exclude the stage if it is not possible to finalize.
    bool possibleToFinalize() const;

    const ColoredInterface& getInterface() const { return interface; }
    std::size_t countChildren();
    const std::vector<Next>& getChildrenHandles() {
        guard();
        return nexts;
    }
    NextFinalizeSlot& getChildFinalizeSlot(std::size_t key);
    template<typename Op>
    const NextOpSlot<Op>& getChildSlot(std::size_t key) {
        guard();
        return nextOpStores.get<Op>().getSlot(key);
    }
    Node getChild(Next next);
    std::string description() const;
};

} // namespace kas
