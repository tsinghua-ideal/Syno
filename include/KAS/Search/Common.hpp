#pragma once

#include <atomic>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Transforms/Transforms.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

struct DesiredSize {
    Size value;
    bool isUnordered;
    operator const Size&() const { return value; }
};

struct CurrentSize {
    Size value;
    int remainingLength;
    operator const Size&() const { return value; }
};

struct CurrentDimension {
    Dimension value;
    int remainingLength;
    operator CurrentSize() const {
        return { value.size(), remainingLength }; 
    }
    operator const Dimension&() const { return value; }
};

struct Next {
    enum class Type: std::uint8_t {
        // Root -> ViewStage
        Reduce,
        // ViewStage -> ViewStage, i.e., views
        Expand,
        Shift,
        Stride,
        Split,
        Unfold,
        Merge,
        // ContractionStage
        Contraction,
        // ViewStage -> FinalStage
        Finalize,
    };
    static constexpr std::size_t NumTypes = 9;
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

    template<GeneralizedOp Op>
    static Next FromOp(const Op *op) {
        return { TypeOf<Op>(), op->opHash() };
    }
    template<typename Op>
    requires(!PrimitiveOpImpl<Op> && std::derived_from<Op, PrimitiveOp>)
    static Next FromOp(const Op *op) {
        return { TypeOf(op->getType()), op->opHash() };
    }

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

    template<GeneralizedOp Op>
    static constexpr Type TypeOf() {
        if constexpr (std::same_as<Op, ReduceOp>) { return Type::Reduce; }
        else if constexpr (std::same_as<Op, ExpandOp>) { return Type::Expand; }
        else if constexpr (std::same_as<Op, ShiftOp>) { return Type::Shift; }
        else if constexpr (std::same_as<Op, StrideOp>) { return Type::Stride; }
        else if constexpr (std::same_as<Op, SplitOp>) { return Type::Split; }
        else if constexpr (std::same_as<Op, UnfoldOp>) { return Type::Unfold; }
        else if constexpr (std::same_as<Op, MergeOp>) { return Type::Merge; }
        else if constexpr (std::same_as<Op, ContractionOp>) { return Type::Contraction; }
        else { static_assert(sizeof(Op) == 0, "Unknown type of Op."); }
    }
    static constexpr Type TypeOf(DimensionType t) {
        switch (t) {
        case DimensionType::Reduce: return Type::Reduce;
        case DimensionType::Expand: return Type::Expand;
        case DimensionType::Shift: return Type::Shift;
        case DimensionType::Stride: return Type::Stride;
        case DimensionType::Split: return Type::Split;
        case DimensionType::Unfold: return Type::Unfold;
        case DimensionType::Merge: return Type::Merge;
        case DimensionType::Share: KAS_CRITICAL("ShareOp must be used indirectly in ContractionOp!");
        default: KAS_CRITICAL("Unknown type of DimensionType.");
        }
    }
};

class Sampler;
class FinalizeOp;

class Arc {
    const Sampler *sampler;
    std::variant<const PrimitiveOp *, const ContractionOp *, const FinalizeOp *> inner;
public:
    Arc(const Sampler *sampler, const PrimitiveOp *op):
        sampler { sampler },
        inner { op } {}
    Arc(const Sampler *sampler, const ContractionOp *op):
        sampler { sampler },
        inner { op } {}
    Arc(const Sampler *sampler, const FinalizeOp *op):
        sampler { sampler },
        inner { op } {}

    template<typename R, typename FP, typename FC, typename FF>
    requires
        std::convertible_to<std::invoke_result_t<FP, const PrimitiveOp *>, R> &&
        std::convertible_to<std::invoke_result_t<FC, const ContractionOp *>, R> &&
        std::convertible_to<std::invoke_result_t<FF, const FinalizeOp *>, R>
    R match(FP&& fp, FC&& fc, FF&& ff) const {
        return std::visit([&](auto arg) -> R {
            if constexpr (std::is_same_v<decltype(arg), const PrimitiveOp *>) {
                return std::invoke(fp, arg);
            } else if constexpr (std::is_same_v<decltype(arg), const ContractionOp *>) {
                return std::invoke(fc, arg);
            } else if constexpr (std::is_same_v<decltype(arg), const FinalizeOp *>) {
                return std::invoke(ff, arg);
            } else {
                KAS_UNREACHABLE();
            }
        }, inner);
    }

    template<typename T>
    const T *as() const {
        if constexpr (std::derived_from<T, PrimitiveOp>) {
            return &dynamic_cast<const T&>(*std::get<const PrimitiveOp *>(inner));
        } else {
            return std::get<const T *>(inner);
        }
    }
    template<typename T>
    const T *tryAs() const {
        if constexpr (std::derived_from<T, PrimitiveOp>) {
            if (!std::holds_alternative<const PrimitiveOp *>(inner)) return nullptr;
            return dynamic_cast<const T *>(std::get<const PrimitiveOp *>(inner));
        } else {
            if (!std::holds_alternative<const T *>(inner)) return nullptr;
            return std::get<const T *>(inner);
        }
    }

    bool operator==(const Arc& rhs) const;
    std::size_t hash() const;
    struct Hash {
        std::size_t operator()(const Arc& arc) const {
            return arc.hash();
        }
    };
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

class NormalStage;

struct NextContractionSlot: Next {
    const ContractionOp *op;
    NormalStage *nextStage;
    bool operator==(const NextContractionSlot& rhs) const noexcept {
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
    std::size_t fill(R&& rawStream, F&& builder) {
        std::size_t original = slots.size();
        std::ranges::move(std::views::transform(builder)(rawStream), std::back_inserter(slots));
        std::ranges::sort(slots);
        return slots.size() - original;
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
    template<typename Pred>
    requires std::predicate<Pred, const Slot&>
    std::size_t remove(Pred&& pred) {
        return std::erase_if(slots, std::forward<Pred>(pred));
    }

    template<typename Pred>
    requires std::predicate<Pred, std::size_t>
    std::size_t removeByIndex(Pred&& pred) {
        std::size_t original = slots.size();
        decltype(slots) newSlots;
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (!std::invoke(std::forward<Pred>(pred), i)) {
                newSlots.emplace_back(std::move(slots[i]));
            }
        }
        slots = std::move(newSlots);
        return original - slots.size();
    }

    std::size_t clear() {
        std::size_t original = slots.size();
        slots.clear();
        return original;
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
using NextContractionSlotStore = GenericNextSlotStore<NextContractionSlot>;

// Call the methods in order due to memory order.
struct DepthwiseStatistics {
    static constexpr std::size_t MaxSearchDepth = 20_uz;
    std::atomic<std::size_t> totalNodes;
    std::atomic<std::size_t> expandedNodes;
    std::atomic<std::size_t> finalChildren;
    std::atomic<std::size_t> initialNonFinalChildren;
    std::atomic<std::size_t> removedNonFinalChildren;
    DepthwiseStatistics& createNode() { ++totalNodes; return *this; }
    DepthwiseStatistics& expandNode() { ++expandedNodes; return *this; }
    DepthwiseStatistics& addFinalChildren(std::size_t cnt) { finalChildren += cnt; return *this; }
    DepthwiseStatistics& addNonFinalChildren(std::size_t cnt) { initialNonFinalChildren += cnt; return *this; }
    DepthwiseStatistics& removeNonFinalChildren(std::size_t cnt) { removedNonFinalChildren += cnt; return *this; }
    void instantDestroy() { --totalNodes; }
    void removeEmbededRedundancy() { --expandedNodes; }
    float branchingFactor() const {
        const std::size_t total = expandedNodes.load();
        const std::size_t children = initialNonFinalChildren.load() + finalChildren.load();
        return total == 0 ? 10.0f : static_cast<float>(children) / total; 
    }
    std::string toString() const {
        return fmt::format(
            "{{totalNodes: {}, expandedNodes: {}, finalChildren: {}, initialNonFinalChildren: {}, removedNonFinalChildren: {}}}",
            totalNodes.load(), expandedNodes.load(), finalChildren.load(), initialNonFinalChildren.load(), removedNonFinalChildren.load()
        );
    }
};

// A node has 3 types.
// But actually 4 types exist.
// IF YOU CHANGE THIS YOU MUST REFER TO Node!
enum class NodeType: std::uint8_t {
    Reducing = 0, // Generating reductions.
    Growing = 1, // Now we have generated Reduce's, repeatedly add PrimitiveOp's.
    Final = 2, // Finalization performed.
    Contraction = 3, // Contraction stage.
};

} // namespace kas

template<>
struct fmt::formatter<kas::Next::Type>: formatter<string_view> {
    template<typename FormatContext>
    auto format(kas::Next::Type t, FormatContext& ctx) const {
        string_view name = "Unknown";
        switch (t) {
        using namespace std::string_view_literals;
        case kas::Next::Type::Reduce: name = "Reduce"sv; break;
        case kas::Next::Type::Expand: name = "Expand"sv; break;
        case kas::Next::Type::Shift: name = "Shift"sv; break;
        case kas::Next::Type::Stride: name = "Stride"sv; break;
        case kas::Next::Type::Split: name = "Split"sv; break;
        case kas::Next::Type::Unfold: name = "Unfold"sv; break;
        case kas::Next::Type::Merge: name = "Merge"sv; break;
        case kas::Next::Type::Contraction: name = "Contraction"sv; break;
        case kas::Next::Type::Finalize: name = "Finalize"sv; break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
