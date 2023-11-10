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

template<GeneralizedOp Op>
using GeneralizedOpPointer = const Op *;

template<GeneralizedOp Op>
struct GeneralizedOpHash {
    std::size_t operator()(GeneralizedOpPointer<Op> op) const noexcept {
        return op->opHash();
    }
};

template<GeneralizedOp Op>
struct GeneralizedOpPointeeEqual {
    bool operator()(const GeneralizedOpPointer<Op>& lhs, const GeneralizedOpPointer<Op>& rhs) const noexcept {
        return *lhs == *rhs;
    }
};

template<GeneralizedOp Op>
class GeneralizedOpStore {
    std::unordered_set<GeneralizedOpPointer<Op>, GeneralizedOpHash<Op>, GeneralizedOpPointeeEqual<Op>> store;
    std::pmr::unsynchronized_pool_resource pool { std::pmr::pool_options {
        .max_blocks_per_chunk = Common::MemoryPoolSize,
        .largest_required_pool_block = sizeof(Op),
    }, std::pmr::new_delete_resource() };
    std::pmr::polymorphic_allocator<Op> allocator { &pool };
    std::mutex mutex;
public:
    GeneralizedOpStore() = default;
    GeneralizedOpStore(const GeneralizedOpStore&) = delete;
    GeneralizedOpStore(GeneralizedOpStore&&) = delete;
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
    ~GeneralizedOpStore() {
        std::lock_guard lock { mutex };
        for (auto op: store) {
            auto mutableOp = const_cast<Op *>(op);
            allocator.template delete_object<Op>(mutableOp);
        }
    }
};

template<typename... Ops>
struct GeneralizedOpStores {
    using Primitives = std::tuple<Ops...>;
    std::tuple<GeneralizedOpStore<Ops>...> stores;
    template<PrimitiveOpImpl Op>
    auto& get() {
        return std::get<GeneralizedOpStore<Op>>(stores);
    }
};

struct PrimitiveOpEqual {
    template<PrimitiveOpImpl Op>
    static bool TypedPrimitiveOpEqual(const PrimitiveOp *const& lhs, const PrimitiveOp *const& rhs) noexcept {
        if (rhs->getType() != Op::Type) {
            return false;
        }
        return static_cast<const Op&>(*lhs) == static_cast<const Op&>(*rhs);
    }

    bool operator()(const PrimitiveOp *const& lhs, const PrimitiveOp *const& rhs) const noexcept {
        switch (lhs->getType()) {
        case DimensionType::Reduce: return TypedPrimitiveOpEqual<ReduceOp>(lhs, rhs);
        case DimensionType::Expand: return TypedPrimitiveOpEqual<ExpandOp>(lhs, rhs);
        case DimensionType::Shift: return TypedPrimitiveOpEqual<ShiftOp>(lhs, rhs);
        case DimensionType::Stride: return TypedPrimitiveOpEqual<StrideOp>(lhs, rhs);
        case DimensionType::Split: return TypedPrimitiveOpEqual<SplitOp>(lhs, rhs);
        case DimensionType::Unfold: return TypedPrimitiveOpEqual<UnfoldOp>(lhs, rhs);
        case DimensionType::Merge: return TypedPrimitiveOpEqual<MergeOp>(lhs, rhs);
        case DimensionType::Share: return TypedPrimitiveOpEqual<ShareOp>(lhs, rhs);
        default: KAS_UNREACHABLE("PrimitiveOpEqual applied to unknown Op!");
        }
    }
};

class PrimitiveOpStore {
    // Remember to register the Ops!
    GeneralizedOpStores<
        ReduceOp,
        ExpandOp,
        ShiftOp, StrideOp,
        SplitOp, UnfoldOp,
        MergeOp, ShareOp
    > stores;
public:
    template<PrimitiveOpImpl Op, typename... Args>
    const Op *get(Args&&... args) {
        return stores.get<Op>().get(std::forward<Args>(args)...);
    }
};

class ContractionOpStore: public GeneralizedOpStore<ContractionOp> {};

} // namespace kas
