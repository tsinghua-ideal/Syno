#pragma once

#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include <boost/functional/hash.hpp>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Transforms/Merge.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Transforms/Stride.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Tuple.hpp"


namespace kas {

namespace detail {

template<typename Op>
using Pointer = const Op *;

template<typename Op>
struct PointerEqual {
    bool operator()(const Pointer<Op>& lhs, const Pointer<Op>& rhs) const noexcept {
        return *lhs == *rhs;
    }
};

template<typename Op, typename Hash>
using Store = std::unordered_set<Pointer<Op>, Hash, PointerEqual<Op>>;

template<typename Op>
struct SingleRepeatLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(Pointer<Op> op) const noexcept {
            auto h = op->initialHash();
            boost::hash_combine(h, op->output.get());
            return h;
        }
    };
    Store<Op, Hash> value;
};
template<typename... Ops>
struct RepeatLikeOpDimensionStore {
    using Primitives = std::tuple<Ops...>;
    std::tuple<SingleRepeatLikeOpDimensionStore<Ops>...> stores;
};

template<typename Op>
struct SingleSplitLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(Pointer<Op> op) const noexcept {
            auto h = op->initialHash();
            boost::hash_combine(h, op->outputLhs.get());
            boost::hash_combine(h, op->outputRhs.get());
            return h;
        }
    };
    Store<Op, Hash> value;
};
template<typename... Ops>
struct SplitLikeOpDimensionStore {
    using Primitives = std::tuple<Ops...>;
    std::tuple<SingleSplitLikeOpDimensionStore<Ops>...> stores;
};

template<typename Op>
struct SingleMergeLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(Pointer<Op> op) const noexcept {
            auto h = op->initialHash();
            boost::hash_combine(h, op->output.get());
            boost::hash_combine(h, op->order);
            return h;
        }
    };
    Store<Op, Hash> value;
};
template<typename... Ops>
struct MergeLikeOpDimensionStore {
    using Primitives = std::tuple<Ops...>;
    std::tuple<SingleMergeLikeOpDimensionStore<Ops>...> stores;
};

} // namespace

// Remember to register the Ops!
class DimensionStore {
    detail::RepeatLikeOpDimensionStore<StrideOp> repeatLikes;
    detail::SplitLikeOpDimensionStore<SplitOp> splitLikes;
    detail::MergeLikeOpDimensionStore<MergeOp, ShareOp> mergeLikes;
    template<typename Op> consteval bool isRepeatLike() {
        return TupleHasTypeV<Op, decltype(repeatLikes)::Primitives>;
    }
    template<typename Op> consteval bool isSplitLike() {
        return TupleHasTypeV<Op, decltype(splitLikes)::Primitives>;
    }
    template<typename Op> consteval bool isMergeLike() {
        return TupleHasTypeV<Op, decltype(mergeLikes)::Primitives>;
    }
    template<typename Op> auto getStore() -> decltype(auto)
    requires(
        isRepeatLike<Op>() || isSplitLike<Op>() || isMergeLike<Op>()
    ) {
        if constexpr (isRepeatLike<Op>()) {
            return std::get<detail::SingleRepeatLikeOpDimensionStore<Op>>(repeatLikes.stores).value;
        } else if constexpr (isSplitLike<Op>()) {
            return std::get<detail::SingleSplitLikeOpDimensionStore<Op>>(splitLikes.stores).value;
        } else if constexpr (isMergeLike<Op>()) {
            return std::get<detail::SingleMergeLikeOpDimensionStore<Op>>(mergeLikes.stores).value;
        }
    }
public:
    template<typename Op, typename... Args>
    Dimension get(Args&&... args) {
        auto& store = getStore<Op>();
        static_assert(std::is_same_v<typename std::remove_reference_t<decltype(store)>::key_type, detail::Pointer<Op>>);
        auto op = std::make_unique<Op>(std::forward<Args>(args)...);
        auto [it, inserted] = store.insert(op.get());
        if (!inserted) {
            // Newly allocated op is automatically destroyed.
            return it->second;
        }
        return Dimension(op.release());
    }
    inline ~DimensionStore() {
        auto deleteOp = [](auto&& store) {
            for (auto&& op: store) {
                delete op;
            }
        };
        TupleForEach(repeatLikes, deleteOp);
        TupleForEach(splitLikes, deleteOp);
        TupleForEach(mergeLikes, deleteOp);
    }
};

} // namespace kas
