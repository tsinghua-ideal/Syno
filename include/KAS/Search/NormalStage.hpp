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
        std::size_t operator()(const Dimensions& interface) const noexcept;
        std::size_t operator()(const NormalStage *nStage) const noexcept;
    };
    struct Equal {
        using is_transparent = void;
        bool operator()(const Dimensions& lhs, const Dimensions& rhs) const noexcept;
        bool operator()(const Dimensions& lhs, const NormalStage *rhs) const noexcept;
        bool operator()(const NormalStage *lhs, const Dimensions& rhs) const noexcept;
        bool operator()(const NormalStage *lhs, const NormalStage *rhs) const noexcept;
    };

private:
    std::unordered_set<NormalStage *, Hash, Equal> interfaces;

public:
    NormalStage *find(const Dimensions& interface) const;
    bool insert(NormalStage *nStage);
    ~NormalStageStore();
};

class NormalStage final: public AbstractStage {
    // The interface decides the hash. Other properties are computed.
    Dimensions interface;

    // Lazily generate children.
    bool childrenGenerated = false;
    bool generatingChildren = false;
    // Node pointers. The nodes are lazily computed. We are searching bottom-up, so the children are actually closer to the input.
    NextSlotStore<NextFinalizeSlot> nextFinalizations;
    NextOpStores<ShiftOp, StrideOp, SplitOp, UnfoldOp, MergeOp, ShareOp> nextOpStores;

    NormalStageStore& getNormalStageStore();

    void removeDeadChildrenFromSlots() override;
    void removeAllChildrenFromSlots() override;
    Finalizability checkForFinalizableChildren() const override;

    // This checks whether the nexts are evaluated. If not, it evaluates them.
    void guardGeneratedChildren();

    std::shared_ptr<TensorView> getFinalize(const FinalizeOp *op) const;

    // Apply the Op to obtain NormalStage.
    NormalStage *getNextOp(const PrimitiveOp *op);

    // This is for pruning. We experimentally finalize this stage, and conservatively exclude the stage if it is not possible to finalize.
    bool possibleToFinalizeByExperimenting() const;

    std::size_t uncheckedCountChildren() const;
    std::vector<Next> uncheckedGetChildrenHandles() const;
    std::vector<Arc> uncheckedGetArcs() const;
    const NextFinalizeSlot *getChildFinalizeSlot(std::size_t key) const;
    template<PrimitiveOpImpl Op>
    const NextOpSlot<Op> *getChildSlot(std::size_t key) const {
        return nextOpStores.get<Op>().getSlot(key);
    }

    template<typename F>
    inline auto guarded(F&& f) -> decltype(f()) {
        guardGeneratedChildren();
        return f();
    }

    template<typename R, typename FP, typename FF>
    requires
        std::convertible_to<std::invoke_result_t<FF, const NextFinalizeSlot&>, R> &&
        std::convertible_to<std::invoke_result_t<FP, const NextOpSlot<ShiftOp>&>, R> &&
        std::convertible_to<std::invoke_result_t<FP, const NextOpSlot<StrideOp>&>, R> &&
        std::convertible_to<std::invoke_result_t<FP, const NextOpSlot<SplitOp>&>, R> &&
        std::convertible_to<std::invoke_result_t<FP, const NextOpSlot<UnfoldOp>&>, R> &&
        std::convertible_to<std::invoke_result_t<FP, const NextOpSlot<MergeOp>&>, R> &&
        std::convertible_to<std::invoke_result_t<FP, const NextOpSlot<ShareOp>&>, R>
    std::optional<R> matchNext(Next next, FP&& fp, FF&& ff) const {
        const auto& [type, key] = next;
        auto box = [&](auto slot) -> std::optional<R> {
            if (!slot) { return std::nullopt; }
            return std::optional<R>(std::in_place, std::invoke(std::forward<FP>(fp), *slot));
        };
        switch (type) {
        case Next::Type::Shift: return box(getChildSlot<ShiftOp>(key));
        case Next::Type::Stride: return box(getChildSlot<StrideOp>(key));
        case Next::Type::Split: return box(getChildSlot<SplitOp>(key));
        case Next::Type::Unfold: return box(getChildSlot<UnfoldOp>(key));
        case Next::Type::Merge: return box(getChildSlot<MergeOp>(key));
        case Next::Type::Share: return box(getChildSlot<ShareOp>(key));
        case Next::Type::Finalize: {
            auto res = getChildFinalizeSlot(key);
            if (!res) return std::nullopt;
            return std::optional<R>(std::in_place, std::invoke(std::forward<FF>(ff), *res));
        }
        default: KAS_UNREACHABLE("Invalid Next {}.", next.toString());
        }
    }

public:
    KAS_STATISTICS_DEF(
        ChildrenFinalize,
        FinalizabilityCheckInvocations,
        TooManyWeights,
        ShapeDeviatesTooMuch,
    );

    NormalStage(Dimensions&& interface, AbstractStage& creator, std::optional<Next::Type> deltaOp);

    const Dimensions& getInterface() const { return interface; }

    std::size_t hash() const override { return NormalStageStore::Hash{}(interface); }
    std::size_t countChildren() override;
    std::vector<Next> getChildrenHandles() override;
    std::vector<Arc> getArcs() override;
    std::optional<Arc> getArcFromHandle(Next next) override;
    std::optional<Node> getChild(Next next) override;
    Node getChild(Arc arc) override;
    std::string getChildDescription(Arc arc) override;
    std::string description() const override;
};

} // namespace kas
