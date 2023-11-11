#pragma once

#include <memory>
#include <memory_resource>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/PrimitiveOp.hpp"
#include "KAS/Transforms/Transforms.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Hash.hpp"
#include "KAS/Utils/Tuple.hpp"


namespace kas {

template<OperationImpl Op>
using OperationPointer = const Op *;

template<OperationImpl Op>
struct OperationHash {
    std::size_t operator()(OperationPointer<Op> op) const noexcept {
        return op->opHash();
    }
};

template<OperationImpl Op>
struct OperationPointeeEqual {
    bool operator()(const OperationPointer<Op>& lhs, const OperationPointer<Op>& rhs) const noexcept {
        return *lhs == *rhs;
    }
};

template<OperationImpl Op>
class OpStore {
    std::unordered_set<OperationPointer<Op>, OperationHash<Op>, OperationPointeeEqual<Op>> store;
    std::pmr::unsynchronized_pool_resource pool { std::pmr::pool_options {
        .max_blocks_per_chunk = Common::MemoryPoolSize,
        .largest_required_pool_block = sizeof(Op),
    }, std::pmr::new_delete_resource() };
    std::pmr::polymorphic_allocator<Op> allocator { &pool };
    std::mutex mutex;
public:
    OpStore() = default;
    OpStore(const OpStore&) = delete;
    OpStore(OpStore&&) = delete;
    template<typename... Args>
    const Op *get(Args&&... args) {
        // Critical section here!
        std::lock_guard lock { mutex };
        auto op = allocator.template new_object<Op>(std::forward<Args>(args)...);
        bool keep = false;
        KAS_DEFER {
            if (!keep) {
                allocator.template delete_object<Op>(op);
            }
        };
        auto [it, inserted] = store.insert(op);
        if (inserted) {
            keep = true;
            return op;
        }
        // Newly allocated op automatically deleted by the deferred block.
        return *it;
    }
    ~OpStore() {
        std::lock_guard lock { mutex };
        for (auto op: store) {
            auto mutableOp = const_cast<Op *>(op);
            allocator.template delete_object<Op>(mutableOp);
        }
    }
};

template<typename... Ops>
struct OpStores {
    using Primitives = std::tuple<Ops...>;
    std::tuple<OpStore<Ops>...> stores;
    template<OperationImpl Op>
    auto& get() {
        return std::get<OpStore<Op>>(stores);
    }
};

class OperationStore {
    // Remember to register the Ops!
    OpStores<
        ReduceOp,
        ExpandOp,
        ShiftOp, StrideOp,
        SplitOp, UnfoldOp,
        MergeOp, ShareOp,
        ContractionOp
    > stores;
public:
    template<OperationImpl Op, typename... Args>
    const Op *get(Args&&... args) {
        return stores.get<Op>().get(std::forward<Args>(args)...);
    }
};

} // namespace kas
