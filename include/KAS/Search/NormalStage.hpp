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
    DimensionStore dimensionStore;
    std::unordered_set<NormalStage *, Hash, Equal> interfaces;

public:
    DimensionStore& dimStore() { return dimensionStore; }
    NormalStage *find(const ColoredInterface& interface) const;
    bool insert(NormalStage *nStage);
    ~NormalStageStore();
};

class NormalStage {
public:
    template<typename Op>
    struct NextOpSlot: NextSlot<Next::TypeOf<Op>()> {
        const Op *op;
        NormalStage *nextStage;
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
        template<typename F>
        requires std::conjunction_v<std::is_invocable<F, NextOpStore<Ops>&>...>
        void forEach(F&& f) {
            std::apply([&f](auto&... store) {
                (f(store), ...);
            }, stores);
        }
        template<typename F>
        requires std::conjunction_v<std::is_invocable<F, NextOpStore<Ops>&>...>
        auto heterogeneousMap(F&& f) {
            return std::apply([&f](auto&... store) {
                return std::tuple<std::invoke_result_t<F, Ops>...> { f(store)... };
            }, stores);
        }
        template<typename F, typename R = std::invoke_result_t<F, NextOpStore<std::tuple_element_t<0, std::tuple<Ops...>>>&>>
        requires
            std::conjunction_v<std::is_invocable<F, NextOpStore<Ops>&>...> &&
            std::conjunction_v<std::is_same<R, std::invoke_result_t<F, NextOpStore<Ops>&>>...>
        auto homogeneousMap(F&& f) {
            std::vector<R> results;
            results.reserve(sizeof...(Ops));
            std::apply([&f, &results](auto&... store) {
                (results.emplace_back(f(store)), ...);
            }, stores);
            return results;
        }
        std::size_t size() const {
            return std::apply([](const auto&... store) {
                return (store.size() + ...);
            }, stores);
        }
        std::vector<Next> toNexts() const {
            auto results = const_cast<NextOpStores<Ops...> *>(this)->homogeneousMap([](const auto& store) { return store.toNexts(); });
            std::vector<Next> flattened;
            std::ranges::move(results | std::views::join, std::back_inserter(flattened));
            return flattened;
        }
    };

    enum class Finalizability {
        Maybe, // Default state.
        Yes, // Determined by expanding subtrees, and see if there are final nodes.
        No, // Determined by expanding all subtrees or conservative experiments.
    };

private:
    // The interface decides the hash. Other properties are computed.
    ColoredInterface interface;

    // Lazily generate children.
    bool childrenGenerated = false;
    // Node pointers. The nodes are lazily computed. We are searching bottom-up, so the children are actually closer to the input.
    NextSlotStore<NextFinalizeSlot> nextFinalizations;
    NextOpStores<ShiftOp, StrideOp, SplitOp, UnfoldOp, MergeOp, ShareOp> nextOpStores;

    // Metadata.
    Sampler& sampler;
    NormalStageStore& getNormalStageStore();
    std::size_t depth; // Stages with identical interfaces must be of the same depth.
    std::array<int, Next::NumTypes> existingOps {};

    Finalizability finalizability = Finalizability::Maybe;
    void determineFinalizability(Finalizability yesOrNo);

    void updateFinalizability();

    template<typename Op>
    const int& existingOp() const { return existingOps[static_cast<std::size_t>(Next::TypeOf<Op>())]; }
    template<typename Op>
    int& existingOp() { return const_cast<int&>(const_cast<const NormalStage *>(this)->existingOp<Op>()); }

    std::size_t remainingDepth() const;

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guard();

    // Execute the finalization to obtain TensorView.
    std::shared_ptr<TensorView> getFinalize(std::size_t key) const;

    // Apply the Op to obtain NormalStage.
    template<typename Op>
    NormalStage *getNextOp(const Op *op) {
        NormalStageStore& store = getNormalStageStore();
        auto newInterface = op->applyToInterface(interface);
        if (NormalStage *found = store.find(newInterface); found) {
            return found;
        } else {
            auto tempStage = std::make_unique<NormalStage>(sampler, std::move(newInterface), *this, Next::TypeOf<Op>());
            if(store.insert(tempStage.get())) {
                return tempStage.release();
            } else {
                KAS_CRITICAL("NormalStageStore::insert() failed.");
            }
        }
    }

    // Remove the Op from store, deleting its enclosing dimensions as well.
    void removeOp(const PrimitiveOp *op) {
        // TODO
    }

    // This is for pruning. We experimentally finalize this stage, and conservatively exclude the stage if it is not possible to finalize.
    bool possibleToFinalizeByExperimenting() const;

    std::size_t uncheckedCountChildren() const;
    std::vector<Next> uncheckedGetChildrenHandles() const;
    const NextFinalizeSlot& uncheckedGetChildFinalizeSlot(std::size_t key) const;
    template<typename Op>
    const NextOpSlot<Op>& uncheckedGetChildSlot(std::size_t key) const {
        return nextOpStores.get<Op>().getSlot(key);
    }
    Node uncheckedGetChild(Next next) const;

public:
    NormalStage(Sampler& sampler, const std::vector<const MapReduceOp *>& reductions);
    NormalStage(Sampler& sampler, ColoredInterface&& interface, const NormalStage& old, Next::Type delta);

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
        TooManyWeights,
        ShapeDeviatesTooMuch,
        FinalizabilityMaybe,
        FinalizabilityYes,
        FinalizabilityNo,
    );

    const ColoredInterface& getInterface() const { return interface; }
    std::size_t hash() const { return NormalStageStore::Hash{}(interface); }
    std::size_t countChildren();
    std::vector<Next> getChildrenHandles();
    const NextFinalizeSlot& getChildFinalizeSlot(std::size_t key);
    template<typename Op>
    const NextOpSlot<Op>& getChildSlot(std::size_t key) {
        guard();
        return uncheckedGetChildSlot<Op>(key);
    }
    Node getChild(Next next);
    std::string description() const;
};

} // namespace kas
