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

class NormalStage final: public AbstractStage {
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
        void forEach(F&& f) const {
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
            auto results = const_cast<NextOpStores<Ops...>&>(*this).homogeneousMap([](const auto& store) { return store.toNexts(); });
            std::vector<Next> flattened;
            std::ranges::move(results | std::views::join, std::back_inserter(flattened));
            return flattened;
        }
    };

    KAS_STATISTICS_DEF(
        ChildrenFinalize,
        FinalizabilityCheckInvocations,
        TooManyWeights,
        ShapeDeviatesTooMuch,
    );

private:
    // The interface decides the hash. Other properties are computed.
    ColoredInterface interface;

    // Lazily generate children.
    bool childrenGenerated = false;
    // Node pointers. The nodes are lazily computed. We are searching bottom-up, so the children are actually closer to the input.
    NextSlotStore<NextFinalizeSlot> nextFinalizations;
    NextOpStores<ShiftOp, StrideOp, SplitOp, UnfoldOp, MergeOp, ShareOp> nextOpStores;

    // Metadata.
    NormalStageStore& getNormalStageStore();

    void removeDeadChildrenFromSlots() override;
    void removeAllChildrenFromSlots() override;
    Finalizability checkForFinalizableChildren() const override;

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guardGeneratedChildren();

    // Execute the finalization to obtain TensorView.
    std::shared_ptr<TensorView> getFinalize(std::size_t key) const;

    // Apply the Op to obtain NormalStage.
    template<typename Op>
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
    template<typename Op>
    const NextOpSlot<Op> *getChildSlot(std::size_t key) const {
        return nextOpStores.get<Op>().getSlot(key);
    }
    std::optional<Node> uncheckedGetChild(Next next) const;
    std::optional<std::string> uncheckedGetChildDescription(Next next);

    template<typename F>
    auto guarded(F&& f) -> decltype(f()) {
        guardGeneratedChildren();
        return AbstractStage::guarded(std::forward<F>(f));
    }

public:
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
