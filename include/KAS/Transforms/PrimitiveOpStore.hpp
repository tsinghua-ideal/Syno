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

template<PrimitiveOpImpl Op>
using Pointer = const Op *;

template<PrimitiveOpImpl Op>
struct OpHash {
    std::size_t operator()(Pointer<Op> op) const noexcept {
        return op->opHash();
    }
};

template<PrimitiveOpImpl Op>
struct PointeeEqual {
    bool operator()(const Pointer<Op>& lhs, const Pointer<Op>& rhs) const noexcept {
        return *lhs == *rhs;
    }
};

template<PrimitiveOpImpl Op>
using OpStore = std::unordered_set<Pointer<Op>, OpHash<Op>, PointeeEqual<Op>>;

template<typename... Ops>
struct OpStores {
    using Primitives = std::tuple<Ops...>;
    std::tuple<OpStore<Ops>...> stores;
    template<PrimitiveOpImpl Op>
    auto& get() {
        return std::get<OpStore<Op>>(stores);
    }
};

} // namespace detail

class PrimitiveOpStore {
    // Remember to register the Ops!
    detail::OpStores<
        MapReduceOp,
        ShiftOp, StrideOp,
        SplitOp, UnfoldOp,
        MergeOp, ShareOp
    > stores;
public:
    PrimitiveOpStore() = default;
    PrimitiveOpStore(const PrimitiveOpStore&) = delete;
    PrimitiveOpStore(PrimitiveOpStore&&) = delete;
    template<PrimitiveOpImpl Op, typename... Args>
    const Op *get(Args&&... args) {
        auto& store = stores.get<Op>();
        static_assert(std::is_same_v<typename std::remove_reference_t<decltype(store)>::key_type, detail::Pointer<Op>>);
        auto op = std::make_unique<Op>(std::forward<Args>(args)...);
        auto [it, inserted] = store.insert(op.get());
        if (!inserted) {
            // Newly allocated op is automatically destroyed.
            return *it;
        }
        return op.release();
    }
    ~PrimitiveOpStore() {
        auto deleteOp = [](auto&& store) {
            for (auto&& op: store) {
                delete op;
            }
        };
        TupleForEach(stores.stores, deleteOp);
    }
};

} // namespace kas
