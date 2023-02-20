#pragma once

#include <tuple>
#include <unordered_map>

#include <boost/functional/hash.hpp>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Tuple.hpp"


namespace kas {

template<typename Op>
struct SingleRepeatLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(const Op& op) const noexcept {
            auto h = op.initialHash();
            boost::hash_combine(h, op.output.get());
            return h;
        }
    };
    std::unordered_map<Op, Dimension::PointerType, Hash> value;
};
template<typename... Ops>
struct RepeatLikeOpDimensionStore {
    using Primitives = std::tuple<Ops...>;
    std::tuple<SingleRepeatLikeOpDimensionStore<Ops>...> stores;
};

template<typename Op>
struct SingleSplitLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(const Op& op) const noexcept {
            auto h = op.initialHash();
            boost::hash_combine(h, op.outputLhs.get());
            boost::hash_combine(h, op.outputRhs.get());
            return h;
        }
    };
    std::unordered_map<Op, Dimension::PointerType, Hash> value;
};
template<typename... Ops>
struct SplitLikeOpDimensionStore {
    using Primitives = std::tuple<Ops...>;
    std::tuple<SingleSplitLikeOpDimensionStore<Ops>...> stores;
};

template<typename Op>
struct SingleMergeLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(const Op& op) const noexcept {
            auto h = op.initialHash();
            boost::hash_combine(h, op.output.get());
            boost::hash_combine(h, op.order);
            return h;
        }
    };
    std::unordered_map<Op, Dimension::PointerType, Hash> value;
};
template<typename... Ops>
struct MergeLikeOpDimensionStore {
    using Primitives = std::tuple<Ops...>;
    std::tuple<SingleMergeLikeOpDimensionStore<Ops>...> stores;
};

// Remember to register the Ops!
class DimensionStore {
    RepeatLikeOpDimensionStore<> repeatLikes;
    SplitLikeOpDimensionStore<> splitLikes;
    MergeLikeOpDimensionStore<ShareOp> mergeLikes;
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
            return std::get<SingleRepeatLikeOpDimensionStore<Op>>(repeatLikes.stores).value;
        } else if constexpr (isSplitLike<Op>()) {
            return std::get<SingleSplitLikeOpDimensionStore<Op>>(splitLikes.stores).value;
        } else if constexpr (isMergeLike<Op>()) {
            return std::get<SingleMergeLikeOpDimensionStore<Op>>(mergeLikes.stores).value;
        }
    }
    template<typename Op, typename... Args>
    Dimension GetFromStore(Args&&... args) {
        auto& store = getStore<Op>();
        auto op = static_cast<DimensionImpl *>(new Op(std::forward<Args>(args)...));
        try {
            auto [it, inserted] = store.try_emplace(*op, op);
            if (!inserted) {
                delete op;
                return it->second;
            }
        } catch (...) {
            delete op;
            throw;
        }
        return Dimension(op);
    }
public:
    template<typename Op, typename... Args>
    Dimension get(Args&&... args) {
        auto& store = getStore<Op>();
        auto op = new Op(std::forward<Args>(args)...);
        try {
            auto [it, inserted] = store.try_emplace(*op, op);
            if (!inserted) {
                delete op;
                return it->second;
            }
        } catch (...) {
            delete op;
            throw;
        }
        return Dimension(static_cast<DimensionImpl>(op));
    }
    inline ~DimensionStore() {
        auto deleteOp = [](auto&& store) {
            for (auto&& [op, dim]: store) {
                delete dim;
            }
        };
        TupleForEach(repeatLikes, deleteOp);
        TupleForEach(splitLikes, deleteOp);
        TupleForEach(mergeLikes, deleteOp);
    }
};

} // namespace kas
