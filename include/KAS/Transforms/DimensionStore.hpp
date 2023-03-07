#pragma once

#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Transforms.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Hash.hpp"
#include "KAS/Utils/Tuple.hpp"


namespace kas {

namespace detail {

template<typename Op>
using Pointer = const Op *;

template<typename Op>
struct PointeeEqual {
    bool operator()(const Pointer<Op>& lhs, const Pointer<Op>& rhs) const noexcept {
        return *lhs == *rhs;
    }
};

template<typename Op, typename Hash>
using Store = std::unordered_set<Pointer<Op>, Hash, PointeeEqual<Op>>;

template<typename Op>
struct SingleRepeatLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(Pointer<Op> op) const noexcept {
            auto h = op->initialHash();
            HashCombine(h, op->output.hash());
            return h;
        }
    };
    Store<Op, Hash> value;
};
template<typename... Ops>
struct RepeatLikeOpDimensionStore {
    using Primitives = std::tuple<Ops...>;
    std::tuple<SingleRepeatLikeOpDimensionStore<Ops>...> stores;
    template<typename Op>
    auto& get() {
        return std::get<SingleRepeatLikeOpDimensionStore<Op>>(stores).value;
    }
};

template<typename Op>
struct SingleSplitLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(Pointer<Op> op) const noexcept {
            auto h = op->initialHash();
            HashCombine(h, op->outputLhs.hash());
            HashCombine(h, op->outputRhs.hash());
            return h;
        }
    };
    Store<Op, Hash> value;
};
template<typename... Ops>
struct SplitLikeOpDimensionStore {
    using Primitives = std::tuple<Ops...>;
    std::tuple<SingleSplitLikeOpDimensionStore<Ops>...> stores;
    template<typename Op>
    auto& get() {
        return std::get<SingleSplitLikeOpDimensionStore<Op>>(stores).value;
    }
};

template<typename Op>
struct SingleMergeLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(Pointer<Op> op) const noexcept {
            auto h = op->initialHash();
            HashCombine(h, op->output.hash());
            return h;
        }
    };
    Store<Op, Hash> value;
};
template<typename... Ops>
struct MergeLikeOpDimensionStore {
    using Primitives = std::tuple<Ops...>;
    std::tuple<SingleMergeLikeOpDimensionStore<Ops>...> stores;
    template<typename Op>
    auto& get() {
        return std::get<SingleMergeLikeOpDimensionStore<Op>>(stores).value;
    }
};

// Remember to register the Ops!
using RepeatLikes = RepeatLikeOpDimensionStore<ShiftOp, StrideOp>;
using SplitLikes = SplitLikeOpDimensionStore<SplitOp, UnfoldOp>;
using MergeLikes = MergeLikeOpDimensionStore<MergeOp, ShareOp>;

} // namespace detail

class DimensionStore {
    detail::RepeatLikes repeatLikes;
    detail::SplitLikes splitLikes;
    detail::MergeLikes mergeLikes;
    template<typename Op> consteval static bool isRepeatLike() {
        return TupleHasTypeV<Op, detail::RepeatLikes::Primitives>;
    }
    template<typename Op> consteval static bool isSplitLike() {
        return TupleHasTypeV<Op, detail::SplitLikes::Primitives>;
    }
    template<typename Op> consteval static bool isMergeLike() {
        return TupleHasTypeV<Op, detail::MergeLikes::Primitives>;
    }
    template<typename Op> auto getStore() -> decltype(auto)
    requires(
        isRepeatLike<Op>() || isSplitLike<Op>() || isMergeLike<Op>()
    ) {
        if constexpr (isRepeatLike<Op>()) {
            return repeatLikes.get<Op>();
        } else if constexpr (isSplitLike<Op>()) {
            return splitLikes.get<Op>();
        } else if constexpr (isMergeLike<Op>()) {
            return mergeLikes.get<Op>();
        }
    }
public:
    DimensionStore() = default;
    DimensionStore(const DimensionStore&) = delete;
    DimensionStore(DimensionStore&&) = delete;
    template<typename Op, typename... Args>
    const Op *get(Args&&... args) {
        auto& store = getStore<Op>();
        static_assert(std::is_same_v<typename std::remove_reference_t<decltype(store)>::key_type, detail::Pointer<Op>>);
        auto op = std::make_unique<Op>(std::forward<Args>(args)...);
        auto [it, inserted] = store.insert(op.get());
        if (!inserted) {
            // Newly allocated op is automatically destroyed.
            return *it;
        }
        return op.release();
    }
    inline ~DimensionStore() {
        auto deleteOp = [](auto&& store) {
            for (auto&& op: store.value) {
                delete op;
            }
        };
        TupleForEach(repeatLikes.stores, deleteOp);
        TupleForEach(splitLikes.stores, deleteOp);
        TupleForEach(mergeLikes.stores, deleteOp);
    }
};

} // namespace kas
