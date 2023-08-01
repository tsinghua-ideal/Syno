#pragma once

#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"

namespace kas {

class DimVisitor {
public:
    virtual void visit(const Iterator& dim);
    virtual void visit(const MapReduce& dim);
    virtual void visit(const RepeatLikeOp::Input& dim);
    virtual void visit(const SplitLikeOp::Input& dim);
    virtual void visit(const MergeLikeOp::Input& dim);
};

template<typename Visitor, typename AttributeType>
concept BottomTopDimPropagator =
    std::move_constructible<AttributeType> &&
    requires(
        Visitor v,
        const Iterator& iterator,
        const MapReduce& mapReduce,
        const RepeatLikeOp::Input& repeatLike,
        const SplitLikeOp::Input& splitLike,
        const MergeLikeOp::Input& mergeLike
    ) {
        { v.transform(iterator) } -> std::same_as<AttributeType>;
        { v.transform(mapReduce) } -> std::same_as<AttributeType>;
        { v.transform(repeatLike) } -> std::same_as<AttributeType>;
        { v.transform(splitLike) } -> std::same_as<AttributeType>;
        { v.transform(mergeLike) } -> std::same_as<std::pair<AttributeType, AttributeType>>;
    };

// A CRTP class that do a DFS, while preserving the dependencies in the DAG.
// That is, a SplitLikeOp will not be visited until both of its outputs are visited.
template<typename Derived, typename AttributeType>
class BottomTopDimVisitor: public DimVisitor {
    Derived& derived() { return static_cast<Derived&>(*this); }
    void guard(const Dimension& dim) {
        if (!attributes.contains(dim)) {
            dim.accept(*this);
        }
    }
public:
    // The attributes.
    std::map<Dimension, AttributeType, Dimension::AddressLessThan> attributes;
    BottomTopDimVisitor() {
        static_assert(BottomTopDimPropagator<Derived, AttributeType>);
    }
    void visit(const Iterator& dim) final {
        auto [_, isNewElem] = attributes.try_emplace(Dimension(&dim), derived().transform(dim));
        KAS_ASSERT(isNewElem);
    }
    void visit(const MapReduce& dim) final {
        auto [_, isNewElem] = attributes.try_emplace(Dimension(&dim), derived().transform(dim));
        KAS_ASSERT(isNewElem);
    }
    void visit(const RepeatLikeOp::Input& dim) final {
        guard(dim.getOp()->output);
        auto [_, isNewElem] = attributes.try_emplace(Dimension(&dim), derived().transform(dim));
        KAS_ASSERT(isNewElem);
    }
    void visit(const SplitLikeOp::Input& dim) final {
        guard(dim.getOp()->outputLhs);
        guard(dim.getOp()->outputRhs);
        auto [_, isNewElem] = attributes.try_emplace(Dimension(&dim), derived().transform(dim));
        KAS_ASSERT(isNewElem);
    }
    void visit(const MergeLikeOp::Input& dim) final {
        auto op = dim.getOp();
        guard(op->output);
        auto [lhs, rhs] = derived().transform(dim);
        auto [_l, isNewElemL] = attributes.try_emplace(op->getInputL(), std::move(lhs));
        auto [_r, isNewElemR] = attributes.try_emplace(op->getInputR(), std::move(rhs));
        KAS_ASSERT(isNewElemL && isNewElemR);
    }
    void propagate(const Dimension& dim) {
        guard(dim);
    }
    template<DimensionRange R>
    void propagate(R&& dims) {
        for (auto&& dim: dims) {
            guard(dim);
        }
    }
};

} // namespace kas
