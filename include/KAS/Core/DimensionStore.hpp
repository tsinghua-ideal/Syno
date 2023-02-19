#pragma once

#include <tuple>
#include <unordered_map>

#include <boost/functional/hash.hpp>

#include "KAS/Core/DimensionDecl.hpp"
#include "KAS/Core/DimensionImpl.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Utils/Tuple.hpp"


namespace kas {

template<RepeatLikePrimitiveOp Op>
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
    std::tuple<SingleRepeatLikeOpDimensionStore<Ops>...> stores;
};

template<SplitLikePrimitiveOp Op>
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
    std::tuple<SingleSplitLikeOpDimensionStore<Ops>...> stores;
};

template<MergeLikePrimitiveOp Op>
struct SingleMergeLikeOpDimensionStore {
    struct Hash {
        std::size_t operator()(const Op& op) const noexcept {
            auto h = op.initialHash();
            boost::hash_combine(h, op.output.get());
            boost::hash_combine(h, op.firstOrSecond);
            return h;
        }
    };
    std::unordered_map<Op, Dimension::PointerType, Hash> value;
};
template<typename... Ops>
struct MergeLikeOpDimensionStore {
    std::tuple<SingleMergeLikeOpDimensionStore<Ops>...> stores;
};

class DimensionStore {
    RepeatLikeOpDimensionStore<> repeatLikes;
    SplitLikeOpDimensionStore<> splitLikes;
    MergeLikeOpDimensionStore<ShareOp> mergeLikes;
    template<PrimitiveOp Op, typename Store, typename... Args>
    static Dimension GetFromStore(Store& store, Args&&... args) {
        auto op = new DimensionImpl { .alts = Op { std::forward<Args>(args)... } };
        try {
            auto [it, inserted] = store.try_emplace(op->as<Op>(), op);
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
    template<RepeatLikePrimitiveOp Op, typename... Args>
    Dimension get(Args&&... args) {
        auto& [store] = std::get<SingleRepeatLikeOpDimensionStore<Op>>(repeatLikes.stores);
        return GetFromStore<Op>(store, std::forward<Args>(args)...);
    }
    template<SplitLikePrimitiveOp Op, typename... Args>
    Dimension get(Args&&... args) {
        auto& [store] = std::get<SingleSplitLikeOpDimensionStore<Op>>(splitLikes.stores);
        return GetFromStore<Op>(store, std::forward<Args>(args)...);
    }
    template<MergeLikePrimitiveOp Op, typename... Args>
    Dimension get(Args&&... args) {
        auto& [store] = std::get<SingleMergeLikeOpDimensionStore<Op>>(mergeLikes.stores);
        return GetFromStore<Op>(store, std::forward<Args>(args)...);
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
