#pragma once

#include "KAS/Core/Expand.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/Reduce.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

class DimVisitor {
public:
    virtual void visit(const Iterator& dim);
    virtual void visit(const Reduce& dim);
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
        const Reduce& reduce,
        const RepeatLikeOp& repeatLike,
        const SplitLikeOp& splitLike,
        const MergeLikeOp& mergeLike
    ) {
        { v.transform(iterator) } -> std::same_as<AttributeType>;
        { v.transform(reduce) } -> std::same_as<AttributeType>;
        { v.transform(repeatLike) } -> std::same_as<AttributeType>;
        { v.transform(splitLike) } -> std::same_as<AttributeType>;
        { v.transform(mergeLike) } -> std::same_as<std::pair<AttributeType, AttributeType>>;
    };

// A CRTP class that does a DFS, while preserving the dependencies in the DAG.
// That is, a SplitLikeOp will not be visited until both of its outputs are visited.
template<typename Derived, typename AttributeType>
class BottomTopDimVisitor: public DimVisitor {
    Derived& derived() { return static_cast<Derived&>(*this); }
    void guard(const Dimension& dim) {
        if (!attributes.contains(dim)) {
            dim.accept(*this);
        }
    }
    template<typename... Args>
    void assertEmplace(const Dimension& dim, Args&&... args) {
        auto [_, isNewElem] = attributes.try_emplace(dim, std::forward<Args>(args)...);
        KAS_ASSERT(isNewElem);
    }
protected:
    using Super = BottomTopDimVisitor<Derived, AttributeType>;
public:
    // The attributes.
    std::map<Dimension, AttributeType, Dimension::AddressLessThan> attributes;
    BottomTopDimVisitor() {
        static_assert(BottomTopDimPropagator<Derived, AttributeType>);
    }
    AttributeType& at(const Dimension& dim) { return attributes.at(dim); }
    const AttributeType& at(const Dimension& dim) const { return attributes.at(dim); }
    void visit(const Iterator& dim) final override {
        assertEmplace(&dim, derived().transform(dim));
    }
    void visit(const Reduce& dim) final override {
        assertEmplace(&dim, derived().transform(dim));
    }
    void visit(const RepeatLikeOp::Input& dim) final override {
        const auto& op = *dim.getOp();
        guard(op.output);
        assertEmplace(&dim, derived().transform(op));
    }
    void visit(const SplitLikeOp::Input& dim) final override {
        const auto& op = *dim.getOp();
        guard(op.outputLhs);
        guard(op.outputRhs);
        assertEmplace(&dim, derived().transform(op));
    }
    void visit(const MergeLikeOp::Input& dim) final override {
        const auto& op = *dim.getOp();
        guard(op.output);
        auto [lhs, rhs] = derived().transform(op);
        assertEmplace(op.getInputL(), std::move(lhs));
        assertEmplace(op.getInputR(), std::move(rhs));
    }
    void propagate(const Topmost& topmost) {
        for (const Dimension& dim: topmost.getDimensions()) {
            guard(dim);
        }
        for (const Expand *expand: topmost.getExpansions()) {
            guard(expand->output);
        }
    }
};

template<typename Visitor, typename AttributeType>
concept TopBottomDimPropagator =
    std::move_constructible<AttributeType> &&
    requires(
        Visitor v,
        const Dimension& dim,
        const RepeatLikeOp& repeatLike,
        const SplitLikeOp& splitLike,
        const MergeLikeOp& mergeLike
    ) {
        { v.transformInput(dim) } -> std::same_as<AttributeType>;
        { v.transformExpand(dim) } -> std::same_as<AttributeType>;
        { v.transform(repeatLike) } -> std::same_as<AttributeType>;
        { v.transform(splitLike) } -> std::same_as<std::pair<AttributeType, AttributeType>>;
        { v.transform(mergeLike) } -> std::same_as<AttributeType>;
    };

// A CRTP class that does a DFS, while preserving the dependencies in the DAG.
// That is, a MergeLikeOp will not be visited until both of its inputs are visited.
template<typename Derived, typename AttributeType>
class TopBottomDimVisitor: public DimVisitor {
    Derived& derived() { return static_cast<Derived&>(*this); }
    bool contains(const Dimension& dim) {
        return attributes.contains(dim);
    }
    template<typename... Args>
    void assertEmplace(const Dimension& dim, Args&&... args) {
        auto [_, isNewElem] = attributes.try_emplace(dim, std::forward<Args>(args)...);
        KAS_ASSERT(isNewElem);
    }
protected:
    using Super = TopBottomDimVisitor<Derived, AttributeType>;
public:
    // The attributes.
    std::map<Dimension, AttributeType, Dimension::AddressLessThan> attributes;
    TopBottomDimVisitor() {
        static_assert(TopBottomDimPropagator<Derived, AttributeType>);
    }
    AttributeType& at(const Dimension& dim) { return attributes.at(dim); }
    const AttributeType& at(const Dimension& dim) const { return attributes.at(dim); }
    void visit(const Iterator& dim) final override {}
    void visit(const Reduce& dim) final override {}
    void visit(const RepeatLikeOp::Input& dim) final override {
        const auto& op = *dim.getOp();
        assertEmplace(op.output, derived().transform(op));
        op.output.accept(*this);
    }
    void visit(const SplitLikeOp::Input& dim) final override {
        const auto& op = *dim.getOp();
        auto [lhs, rhs] = derived().transform(op);
        assertEmplace(op.outputLhs, std::move(lhs));
        assertEmplace(op.outputRhs, std::move(rhs));
        op.outputLhs.accept(*this);
        op.outputRhs.accept(*this);
    }
    void visit(const MergeLikeOp::Input& dim) final override {
        if (!contains(dim.getOther())) return;
        const auto& op = *dim.getOp();
        assertEmplace(op.output, derived().transform(op));
        op.output.accept(*this);
    }
    void propagate(const Topmost& topmost) {
        for (const Dimension& dim: topmost.getDimensions()) {
            assertEmplace(dim, derived().transformInput(dim));
        }
        for (const Expand *expand: topmost.getExpansions()) {
            assertEmplace(expand->output, derived().transformExpand(expand->output));
        }
        for (const Dimension& dim: topmost.getDimensions()) {
            dim.accept(*this);
        }
        for (const Expand *expand: topmost.getExpansions()) {
            expand->output.accept(*this);
        }
    }
};

} // namespace kas
