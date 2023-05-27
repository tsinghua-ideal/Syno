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
#include "KAS/Utils/Hash.hpp"

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
    Type type;
    // This can be hash, or any arbitrary fixed number, as long as this is invariant between runs.
    std::size_t key;

    // For Python.
    bool operator==(const Next& rhs) const noexcept = default;
    // For Python.
    std::size_t hash() const noexcept {
        std::size_t h = static_cast<std::size_t>(type);
        HashCombine(h, key);
        return h;
    }

    // <type>(<key>)
    std::string toString() const;
    // Detailed descriptions of this Op, based on the node.
    std::string description(const Node& node) const;

    static std::map<Type, std::size_t> CountTypes(const std::vector<Next>& nexts);

    template<typename Op>
    static constexpr Type TypeOf() {
        if constexpr (std::same_as<Op, ShiftOp>) { return Type::Shift; }
        else if constexpr (std::same_as<Op, StrideOp>) { return Type::Stride; }
        else if constexpr (std::same_as<Op, SplitOp>) { return Type::Split; }
        else if constexpr (std::same_as<Op, UnfoldOp>) { return Type::Unfold; }
        else if constexpr (std::same_as<Op, MergeOp>) { return Type::Merge; }
        else if constexpr (std::same_as<Op, ShareOp>) { return Type::Share; }
        else { return static_cast<Type>(-1); }
    }
};

template<Next::Type _SlotType>
struct NextSlot {
    static constexpr Next::Type SlotType = _SlotType;

    // This is the key, used for indexing.
    std::size_t key;

    Next toNext() const {
        return Next { SlotType, key };
    }
};

template<typename Slot>
requires
    std::derived_from<Slot, NextSlot<Slot::SlotType>>
    && std::move_constructible<Slot>
class NextSlotStore {
    using Self = NextSlotStore<Slot>;

    std::vector<Slot> slots;

public:
    NextSlotStore() = default;
    NextSlotStore(const NextSlotStore&) = delete;
    NextSlotStore(NextSlotStore&&) = delete;

    std::size_t size() const { return slots.size(); }

    // A slot can be found by its key, because we have sorted the slots by key.
    std::vector<Slot>::const_iterator findSlot(std::size_t key) const {
        return std::ranges::lower_bound(slots, key, std::less{}, &Slot::key);;
    }
    std::vector<Slot>::iterator findSlot(std::size_t key) {
        return std::ranges::lower_bound(slots, key, std::less{}, &Slot::key);;
    }
    const Slot& getSlot(std::size_t key) const {
        auto it = findSlot(key);
        KAS_ASSERT(it != slots.end() && it->key == key, "Slot not found.");
        return *it;
    }
    Slot& getSlot(std::size_t key) { return const_cast<Slot&>(std::as_const(*this).getSlot(key)); }

    // Fill the slots with the the given range, and then sort by key.
    template<std::ranges::input_range R, typename F>
    requires std::convertible_to<std::invoke_result_t<F, std::ranges::range_reference_t<R>>, Slot>
    Self& fill(R&& rawStream, F&& builder) {
        std::ranges::move(std::views::transform(builder)(rawStream), std::back_inserter(slots));
        std::ranges::sort(slots, std::less{}, &Slot::key);
        return *this;
    }

    // Add a new slot to the end of slots.
    Self& append(auto&& slot) {
        slots.emplace_back(std::forward<decltype(slot)>(slot));
        if (slots.size() > 1) {
            KAS_ASSERT(slots.back().key >= slots[slots.size() - 2].key, "Slots must be sorted by key.");
        }
        return *this;
    }

    // Remove all slots that satisfy the predicate.
    template<typename Pred, typename Callback>
    requires std::predicate<Pred, Slot> && std::invocable<Callback, Slot>
    Self& remove(Pred&& pred, Callback&& callback) {
        auto [first, last] = std::ranges::remove_if(slots, std::forward<Pred>(pred));
        std::ranges::for_each(first, last, std::forward<Callback>(callback));
        slots.erase(first, last);
        return *this;
    }
    template<typename Pred>
    requires std::predicate<Pred, Slot>
    Self& remove(Pred&& pred) {
        return remove(std::forward<Pred>(pred), [](const Slot&){});
    }

    std::vector<Next> toNexts() const {
        std::vector<Next> nexts;
        std::ranges::move(slots | std::views::transform(&Slot::toNext), std::back_inserter(nexts));
        return nexts;
    }
};

class TensorView;
class Sampler;
class ReductionStage;
class Stage;

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
    std::variant<ReductionStage *, Stage *, std::shared_ptr<TensorView> > inner;
    Type type() const noexcept {
        return static_cast<Type>(inner.index());
    }
    template<typename R, typename FR, typename FG, typename FF>
    requires
        std::convertible_to<std::invoke_result_t<FR, ReductionStage *>, R> &&
        std::convertible_to<std::invoke_result_t<FG, Stage *>, R> &&
        std::convertible_to<std::invoke_result_t<FF, std::shared_ptr<TensorView> >, R>
    R match(FR&& fr, FG&& fg, FF&& ff) const {
        return std::visit([&](auto arg) -> R {
            if constexpr (std::is_same_v<decltype(arg), ReductionStage *>) {
                return fr(arg);
            } else if constexpr (std::is_same_v<decltype(arg), Stage *>) {
                return fg(arg);
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
    Node(Sampler *sampler, Stage *stage):
        sampler { sampler }, inner { stage } {}
    Node(Sampler *sampler, std::shared_ptr<TensorView> kernel):
        sampler { sampler }, inner { kernel } {}

    // For Python.
    bool operator==(const Node& rhs) const noexcept = default;
    // For Python.
    std::size_t hash() const noexcept {
        return std::hash<decltype(inner)>{}(inner);
    }

    std::shared_ptr<TensorView> asFinal() const;
    std::unique_ptr<Kernel> realizeAsFinal(const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options) const;
    // Obtain the mappings from Sampler, and do not solve the paddings. We only want to estimate the FLOPs.
    std::size_t estimateTotalFLOPsAsFinal() const;

    // The count of children nodes.
    std::size_t countChildren() const;
    std::vector<Next> getChildrenHandles() const;
    Node getChild(Next next) const;
    bool isFinal() const { return type() == Type::Final; }
    std::string toString() const;
};

}

template<>
struct fmt::formatter<kas::Next::Type>: formatter<string_view> {
    template<typename FormatContext>
    auto format(kas::Next::Type t, FormatContext& ctx) const {
        string_view name = "Unknown";
        switch (t) {
        using namespace std::literals;
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
