#pragma once

#include <cstddef>
#include <map>
#include <ranges>
#include <string>
#include <type_traits>
#include <variant>

#include <fmt/core.h>

#include <KAS/Transforms.hpp>
#include "KAS/CodeGen/Kernel.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Utils/Hash.hpp"
#include "KAS/Utils/Ranges.hpp"

namespace kas {

class Node;

struct Next {
    enum class Type: std::uint8_t {
        // Root -> Growing
        MapReduce,
        // Growing -> Growing, i.e., PrimitiveOp's
        Shift,
        Stride,
        Split,
        Unfold,
        Merge,
        Share,
        // Growing -> Final
        Finalize,
    };
    static constexpr std::size_t NumTypes = 8;
    using OpCounterType = std::uint8_t;
    struct OpTypeCounter: std::array<OpCounterType, Next::NumTypes> {
        const OpCounterType& operator[](Type t) const noexcept {
            return std::array<OpCounterType, Next::NumTypes>::operator[](static_cast<std::size_t>(t));
        }
        OpCounterType& operator[](Type t) noexcept {
            return const_cast<OpCounterType&>(std::as_const(*this)[t]);
        }
    };
    Type type;
    // This can be hash, or any arbitrary fixed number, as long as this is invariant between runs.
    std::size_t key;

    // For Python.
    bool operator==(const Next& rhs) const noexcept = default;
    std::weak_ordering operator<=>(const Next& rhs) const noexcept = default;
    // For Python.
    std::size_t hash() const noexcept {
        using namespace std::string_view_literals;
        static const auto nextHash = std::hash<std::string_view>{}("Next"sv);
        auto h = nextHash;
        HashCombine(h, type);
        HashCombine(h, key);
        return h;
    }

    // <type>(<key>)
    std::string toString() const;

    static std::map<Type, std::size_t> CountTypes(const std::vector<Next>& nexts);

    template<PrimitiveOpImpl Op>
    static constexpr Type TypeOf() {
        if constexpr (std::same_as<Op, MapReduceOp>) { return Type::MapReduce; }
        else if constexpr (std::same_as<Op, ShiftOp>) { return Type::Shift; }
        else if constexpr (std::same_as<Op, StrideOp>) { return Type::Stride; }
        else if constexpr (std::same_as<Op, SplitOp>) { return Type::Split; }
        else if constexpr (std::same_as<Op, UnfoldOp>) { return Type::Unfold; }
        else if constexpr (std::same_as<Op, MergeOp>) { return Type::Merge; }
        else if constexpr (std::same_as<Op, ShareOp>) { return Type::Share; }
        else { KAS_CRITICAL("Unknown type of Op."); }
    }
    static constexpr Type TypeOf(DimensionType t) {
        switch (t) {
        case DimensionType::MapReduce: return Type::MapReduce;
        case DimensionType::Shift: return Type::Shift;
        case DimensionType::Stride: return Type::Stride;
        case DimensionType::Split: return Type::Split;
        case DimensionType::Unfold: return Type::Unfold;
        case DimensionType::Merge: return Type::Merge;
        case DimensionType::Share: return Type::Share;
        default: KAS_CRITICAL("Unknown type of DimensionType.");
        }
    }
};

class Sampler;
class FinalizeOp;

class Arc {
    const Sampler *sampler;
    std::variant<const PrimitiveOp *, const FinalizeOp *> inner;
public:
    Arc(const Sampler *sampler, const PrimitiveOp *op):
        sampler { sampler },
        inner { op } {}
    Arc(const Sampler *sampler, const FinalizeOp *op):
        sampler { sampler },
        inner { op } {}

    template<typename R, typename FP, typename FF>
    requires
        std::convertible_to<std::invoke_result_t<FP, const PrimitiveOp *>, R> &&
        std::convertible_to<std::invoke_result_t<FF, const FinalizeOp *>, R>
    R match(FP&& fp, FF&& ff) const {
        return std::visit([&](auto arg) -> R {
            if constexpr (std::is_same_v<decltype(arg), const PrimitiveOp *>) {
                return fp(arg);
            } else if constexpr (std::is_same_v<decltype(arg), const FinalizeOp *>) {
                return ff(arg);
            } else {
                KAS_UNREACHABLE();
            }
        }, inner);
    }

    template<typename T>
    const T *as() const {
        return std::get<const T *>(inner);
    }
    template<typename T>
    const T *tryAs() const {
        return std::holds_alternative<const T *>(inner) ? std::get<const T *>(inner) : nullptr;
    }

    bool operator==(const Arc& rhs) const;
    std::size_t hash() const;
    Next toNext() const;
    std::string toString() const;
};

class AbstractStage;

struct NextStageSlot: Next {
    const PrimitiveOp *op;
    AbstractStage *nextStage;
    bool operator==(const NextStageSlot& rhs) const noexcept {
        // This is enough for equality.
        return op == rhs.op;
    }
    // Compare the slots as Next. That is, first compare the type, then compare the hash.
    std::weak_ordering operator<=>(const NextStageSlot& rhs) const noexcept {
        return static_cast<const Next&>(*this) <=> static_cast<const Next&>(rhs);
    }
    static std::size_t GetKey(const PrimitiveOp *op) { return op->opHash(); }
    Arc toArc(const Sampler *sampler) const { return Arc(sampler, op); }
};

template<typename Slot>
class GenericNextSlotStore {
    std::vector<Slot> slots;

public:
    GenericNextSlotStore() = default;
    GenericNextSlotStore(const GenericNextSlotStore&) = delete;
    GenericNextSlotStore(GenericNextSlotStore&&) = delete;

    const std::vector<Slot>& getRawSlots() const { return slots; }

    std::size_t size() const { return slots.size(); }

    // A slot can be found by its type and key, because we have sorted the slots by type and key.
    std::vector<Slot>::const_iterator findSlot(Next next) const {
        return std::ranges::lower_bound(slots, next);
    }
    std::vector<Slot>::iterator findSlot(Next next) {
        return std::ranges::lower_bound(slots, next);
    }
    const Slot *getSlot(Next next) const {
        auto it = findSlot(next);
        if (it == slots.end() || next != *it) {
            return nullptr;
        }
        return &*it;
    }
    Slot *getSlot(Next next) { return const_cast<Slot *>(std::as_const(*this).getSlot(next)); }

    template<typename R, typename F>
    requires std::convertible_to<std::invoke_result_t<F, const Slot&>, R>
    std::optional<R> findTransform(Next next, F&& f) const {
        auto ptr = getSlot(next);
        if (!ptr) return std::nullopt;
        return std::optional<R>(std::in_place, std::invoke(f, *ptr));
    }

    // Fill the slots with the the given range, and then sort by key.
    template<std::ranges::input_range R, typename F>
    requires std::convertible_to<std::invoke_result_t<F, std::ranges::range_reference_t<R>>, Slot>
    GenericNextSlotStore& fill(R&& rawStream, F&& builder) {
        std::ranges::move(std::views::transform(builder)(rawStream), std::back_inserter(slots));
        std::ranges::sort(slots);
        return *this;
    }

    // Add a new slot to the end of slots.
    GenericNextSlotStore& append(auto&& slot) {
        slots.emplace_back(std::forward<decltype(slot)>(slot));
        if (slots.size() > 1) {
            KAS_ASSERT(slots.back().key >= slots[slots.size() - 2].key, "Slots must be sorted by key.");
        }
        return *this;
    }

    // Remove all slots that satisfy the predicate.
    template<typename Pred, typename Callback>
    requires std::predicate<Pred, const Slot&> && std::invocable<Callback, const Slot&>
    GenericNextSlotStore& remove(Pred&& pred, Callback&& callback) {
        auto [first, last] = std::ranges::remove_if(slots, std::forward<Pred>(pred));
        std::ranges::for_each(first, last, std::forward<Callback>(callback));
        slots.erase(first, last);
        return *this;
    }
    template<typename Pred>
    requires std::predicate<Pred, const Slot&>
    GenericNextSlotStore& remove(Pred&& pred) {
        return remove(std::forward<Pred>(pred), [](const Slot&){});
    }

    template<typename Pred>
    requires std::predicate<Pred, std::size_t>
    GenericNextSlotStore& removeByIndex(Pred&& pred) {
        decltype(slots) newSlots;
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (!std::invoke(std::forward<Pred>(pred), i)) {
                newSlots.emplace_back(std::move(slots[i]));
            }
        }
        slots = std::move(newSlots);
        return *this;
    }

    GenericNextSlotStore& clear() {
        slots.clear();
        return *this;
    }

    template<typename F>
    requires std::invocable<F, Slot&>
    GenericNextSlotStore& forEach(F&& f) {
        std::ranges::for_each(slots, std::forward<F>(f));
        return *this;
    }
    template<typename F>
    requires std::invocable<F, const Slot&>
    const GenericNextSlotStore& forEach(F&& f) const {
        std::ranges::for_each(slots, std::forward<F>(f));
        return *this;
    }

    template<typename F>
    requires std::invocable<F, const Slot&>
    auto map(F&& f) const -> std::vector<std::decay_t<std::invoke_result_t<F, const Slot&>>> {
        return ranges::to<std::vector<std::decay_t<std::invoke_result_t<F, const Slot&>>>>(std::views::transform(slots, std::forward<F>(f)));
    }

    std::vector<Next> toNexts() const {
        std::vector<Next> nexts;
        std::ranges::copy(slots, std::back_inserter(nexts));
        return nexts;
    }

    std::vector<Arc> toArcs(const Sampler *sampler) const {
        std::vector<Arc> arcs;
        std::ranges::move(
            slots
            | std::views::transform([=](const Slot& slot) {
                return slot.toArc(sampler);
            }),
            std::back_inserter(arcs)
        );
        return arcs;
    }

    void checkHashCollisionAndRemove() {
        if (auto it = std::ranges::adjacent_find(slots); it != slots.end()) {
            KAS_WARNING("Hash collision {} detected. Now removing.", it->toString());
        } else {
            return;
        }
        // In case of duplicate keys, remove all of them.
        // This is because we don't want to have duplicate keys in the final result.
        // This is a very rare case, so we don't care about the performance.
        std::map<std::size_t, std::vector<Slot *>> map;
        for (auto& slot: slots) {
            map[slot.key].emplace_back(&slot);
        }
        std::vector<Slot> newSlots;
        for (auto& [key, slots]: map) {
            if (slots.size() == 1) {
                newSlots.emplace_back(std::move(*slots[0]));
            }
        }
        slots = std::move(newSlots);
    }
};
using NextSlotStore = GenericNextSlotStore<NextStageSlot>;

class AbstractStage;
class ReductionStage;
class NormalStage;
class TensorView;

class Node {
    friend struct Next;

    Sampler *sampler;

    // A node has 3 types.
    enum class Type: std::uint8_t {
        Reducing = 0, // Generating reductions.
        Growing = 1, // Now we have generated MapReduce's, repeatedly add PrimitiveOp's.
        Final = 2, // Finalization performed.
    };
    // This corresponds to the three types.
    std::variant<ReductionStage *, NormalStage *, std::shared_ptr<TensorView>> inner;
    Type type() const noexcept {
        return static_cast<Type>(inner.index());
    }
    template<typename R, typename FR, typename FN, typename FF>
    requires
        std::convertible_to<std::invoke_result_t<FR, ReductionStage *>, R> &&
        std::convertible_to<std::invoke_result_t<FN, NormalStage *>, R> &&
        std::convertible_to<std::invoke_result_t<FF, std::shared_ptr<TensorView> >, R>
    R match(FR&& fr, FN&& fn, FF&& ff) const {
        return std::visit([&](auto arg) -> R {
            if constexpr (std::is_same_v<decltype(arg), ReductionStage *>) {
                return fr(arg);
            } else if constexpr (std::is_same_v<decltype(arg), NormalStage *>) {
                return fn(arg);
            } else if constexpr (std::is_same_v<decltype(arg), std::shared_ptr<TensorView> >) {
                return ff(arg);
            } else {
                KAS_UNREACHABLE();
            }
        }, inner);
    }
    template<typename R, typename FS, typename FF>
    requires
        std::convertible_to<std::invoke_result_t<FS, AbstractStage *>, R> &&
        std::convertible_to<std::invoke_result_t<FF, std::shared_ptr<TensorView> >, R>
    R match(FS&& fs, FF&& ff) const {
        return std::visit([&](auto arg) -> R {
            if constexpr (std::is_same_v<decltype(arg), ReductionStage *>) {
                return fs(arg);
            } else if constexpr (std::is_same_v<decltype(arg), NormalStage *>) {
                return fs(arg);
            } else if constexpr (std::is_same_v<decltype(arg), std::shared_ptr<TensorView> >) {
                return ff(arg);
            } else {
                KAS_UNREACHABLE();
            }
        }, inner);
    }

public:
    Node(Sampler *sampler, ReductionStage *rStage):
        sampler { sampler }, inner { rStage } {}
    Node(Sampler *sampler, NormalStage *nStage):
        sampler { sampler }, inner { nStage } {}
    Node(Sampler *sampler, std::shared_ptr<TensorView> kernel):
        sampler { sampler }, inner { kernel } {}

    // For Python.
    bool operator==(const Node& rhs) const;
    // For Python.
    std::size_t hash() const;

    AbstractStage *tryAsStage() const;
    NormalStage *asNormalStage() const;
    std::shared_ptr<TensorView> asFinal() const;
    std::unique_ptr<Kernel> realizeAsFinal(const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options) const;
    // Obtain the mappings from Sampler, and do not solve the paddings. We only want to estimate the FLOPs.
    std::size_t estimateTotalFLOPsAsFinal() const;
    // No tensors!
    void generateGraphviz(const std::string& dir, const std::string& name) const;
    // With tensors!
    void generateGraphvizAsFinal(const std::string& dir, const std::string& name) const;
    std::string getNestedLoopsAsFinal() const;

    // The count of children nodes.
    std::size_t countChildren() const;
    std::vector<Next> getChildrenHandles() const;
    std::vector<Arc> getChildrenArcs() const;
    std::optional<Arc> getArcFromHandle(Next next) const;
    std::optional<Node> getChild(Next next) const;
    bool canAcceptArc(Arc arc) const;
    Node getChildFromArc(Arc arc) const;
    std::vector<Next> getPossiblePath() const;
    std::vector<Arc> getComposingArcs() const;
    void expand(int layers) const;
    void expandAsync(int layers) const;
    std::optional<std::string> getChildDescription(Next next) const;
    bool isFinal() const { return type() == Type::Final; }
    bool isDeadEnd() const;
    bool discoveredFinalDescendant() const;
    std::string toString() const;
};

}

template<>
struct fmt::formatter<kas::Next::Type>: formatter<string_view> {
    template<typename FormatContext>
    auto format(kas::Next::Type t, FormatContext& ctx) const {
        string_view name = "Unknown";
        switch (t) {
        using namespace std::string_view_literals;
        case kas::Next::Type::MapReduce: name = "MapReduce"sv; break;
        case kas::Next::Type::Shift: name = "Shift"sv; break;
        case kas::Next::Type::Stride: name = "Stride"sv; break;
        case kas::Next::Type::Split: name = "Split"sv; break;
        case kas::Next::Type::Unfold: name = "Unfold"sv; break;
        case kas::Next::Type::Merge: name = "Merge"sv; break;
        case kas::Next::Type::Share: name = "Share"sv; break;
        case kas::Next::Type::Finalize: name = "Finalize"sv; break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
