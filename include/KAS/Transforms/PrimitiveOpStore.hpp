#pragma once

#include <memory>
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

namespace detail {

template <PrimitiveOpImpl Op> using Pointer = const Op *;

template <PrimitiveOpImpl Op> struct OpHash {
  std::size_t operator()(Pointer<Op> op) const noexcept { return op->opHash(); }
};

template <PrimitiveOpImpl Op> struct PointeeEqual {
  bool operator()(const Pointer<Op> &lhs,
                  const Pointer<Op> &rhs) const noexcept {
    return *lhs == *rhs;
  }
};

template <PrimitiveOpImpl Op> struct OpStore {
  std::unordered_set<Pointer<Op>, OpHash<Op>, PointeeEqual<Op>> store;
  std::mutex mutex;
};

template <typename... Ops> struct OpStores {
  using Primitives = std::tuple<Ops...>;
  std::tuple<OpStore<Ops>...> stores;
  template <PrimitiveOpImpl Op> auto &get() {
    return std::get<OpStore<Op>>(stores);
  }
};

} // namespace detail

struct PrimitiveOpEqual {
  template <PrimitiveOpImpl Op>
  static bool TypedPrimitiveOpEqual(const PrimitiveOp *const &lhs,
                                    const PrimitiveOp *const &rhs) noexcept {
    if (rhs->getType() != Op::Type) {
      return false;
    }
    return static_cast<const Op &>(*lhs) == static_cast<const Op &>(*rhs);
  }

  bool operator()(const PrimitiveOp *const &lhs,
                  const PrimitiveOp *const &rhs) const noexcept {
    switch (lhs->getType()) {
    case DimensionType::MapReduce:
      return TypedPrimitiveOpEqual<MapReduceOp>(lhs, rhs);
    case DimensionType::Expand:
      return TypedPrimitiveOpEqual<ExpandOp>(lhs, rhs);
    case DimensionType::Shift:
      return TypedPrimitiveOpEqual<ShiftOp>(lhs, rhs);
    case DimensionType::Stride:
      return TypedPrimitiveOpEqual<StrideOp>(lhs, rhs);
    case DimensionType::Split:
      return TypedPrimitiveOpEqual<SplitOp>(lhs, rhs);
    case DimensionType::Unfold:
      return TypedPrimitiveOpEqual<UnfoldOp>(lhs, rhs);
    case DimensionType::Merge:
      return TypedPrimitiveOpEqual<MergeOp>(lhs, rhs);
    case DimensionType::Share:
      return TypedPrimitiveOpEqual<ShareOp>(lhs, rhs);
    default:
      KAS_UNREACHABLE("PrimitiveOpEqual applied to unknown Op!");
    }
  }
};

class PrimitiveOpStore {
  // Remember to register the Ops!
  detail::OpStores<MapReduceOp, ExpandOp, ShiftOp, StrideOp, SplitOp, UnfoldOp,
                   MergeOp, ShareOp>
      stores;

public:
  PrimitiveOpStore() = default;
  PrimitiveOpStore(const PrimitiveOpStore &) = delete;
  PrimitiveOpStore(PrimitiveOpStore &&) = delete;
  template <PrimitiveOpImpl Op, typename... Args>
  const Op *get(Args &&...args) {
    auto &[store, mutex] = stores.get<Op>();
    static_assert(std::is_same_v<
                  typename std::remove_reference_t<decltype(store)>::key_type,
                  detail::Pointer<Op>>);
    auto op = std::make_unique<Op>(std::forward<Args>(args)...);
    // Critical section here!
    {
      std::lock_guard lock{mutex};
      auto [it, inserted] = store.insert(op.get());
      if (!inserted) {
        // Newly allocated op is automatically destroyed.
        return *it;
      }
      return op.release();
    }
  }
  ~PrimitiveOpStore() {
    auto deleteOp = [](auto &&store) {
      for (auto &&op : store.store) {
        delete op;
      }
    };
    TupleForEach(stores.stores, deleteOp);
  }
};

} // namespace kas
