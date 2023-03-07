#pragma once

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

class DimensionStore;

// There are 3 kinds of `PrimitiveOp`'s, listed below. Those classes can transform `Dimension`s, from those that index the output tensor, to forms that index the original tensors. So this is also kind of bottom-up.

// By repeat-like, we refer to the primitives that have one input iterator and one output iterator.
class RepeatLikeOp {
public:
    class Input: public DimensionImpl {
    protected:
        const RepeatLikeOp *op;
        inline Input(const RepeatLikeOp *op): op { op } {}
        template<typename Derived>
        const Derived *getDerivedOp() const noexcept {
            return static_cast<const Derived *>(op);
        }
    public:
        inline std::size_t hash() const noexcept final override {
            std::size_t h = op->initialHash();
            HashCombine(h, op->output.hash());
            return h;
        }
        inline const RepeatLikeOp *getOp() const noexcept { return op; }
    };
    Dimension output;
    RepeatLikeOp(auto&& output):
        output { std::forward<decltype(output)>(output) }
    {}
    RepeatLikeOp(const RepeatLikeOp&) = delete;
    RepeatLikeOp(RepeatLikeOp&&) = delete;
    virtual DimensionType getType() const noexcept = 0;
    virtual std::size_t initialHash() const noexcept = 0;
    // We would like to store the DimensionImpl inside this class, so we can just return a reference to part of this object.
    virtual Dimension getInput() const = 0;
    virtual IteratorValue value(const IteratorValue& value) const = 0;
    ~RepeatLikeOp() = default;

    Interface applyTo(const Interface& interface) const;
};

// By split-like, we refer to the primitives that have one input iterator and two output iterators.
class SplitLikeOp {
public:
    class Input: public DimensionImpl {
    protected:
        const SplitLikeOp *op;
        inline Input(const SplitLikeOp *op): op { op } {}
        template<typename Derived>
        const Derived *getDerivedOp() const noexcept {
            return static_cast<const Derived *>(op);
        }
    public:
        inline std::size_t hash() const noexcept final override {
            std::size_t h = op->initialHash();
            HashCombine(h, op->outputLhs.hash());
            HashCombine(h, op->outputRhs.hash());
            return h;
        }
        inline const SplitLikeOp *getOp() const noexcept { return op; }
    };
    Dimension outputLhs, outputRhs;
    SplitLikeOp(auto&& outputLhs, auto&& outputRhs):
        outputLhs { std::forward<decltype(outputLhs)>(outputLhs) },
        outputRhs { std::forward<decltype(outputRhs)>(outputRhs) }
    {}
    SplitLikeOp(const SplitLikeOp&) = delete;
    SplitLikeOp(SplitLikeOp&&) = delete;
    virtual DimensionType getType() const noexcept = 0;
    virtual std::size_t initialHash() const noexcept = 0;
    virtual Dimension getInput() const = 0;
    virtual IteratorValue value(const IteratorValue& leftValue, const IteratorValue& rightValue) const = 0;
    ~SplitLikeOp() = default;

    Interface applyTo(const Interface& interface) const;
};

enum class Order: bool {
    Left = false,
    Right = true,
};
// By merge-like, we refer to the primitives that have two input iterators and one output iterator.
class MergeLikeOp {
public:
    class Input: public DimensionImpl {
    protected:
        const MergeLikeOp *op;
        Order order;
        inline Input(const MergeLikeOp *op, Order order): op { op }, order { order } {}
        template<typename Derived>
        const Derived *getDerivedOp() const noexcept {
            return static_cast<const Derived *>(op);
        }
    public:
        inline std::size_t hash() const noexcept final override {
            std::size_t h = op->initialHash();
            HashCombine(h, op->output.hash());
            HashCombine(h, order);
            return h;
        }
        inline const MergeLikeOp *getOp() const noexcept { return op; }
        inline Order getOrder() const noexcept { return order; }
    };
    Dimension output;
    MergeLikeOp(auto&& output):
        output { std::forward<decltype(output)>(output) }
    {}
    MergeLikeOp(const MergeLikeOp&) = delete;
    MergeLikeOp(MergeLikeOp&&) = delete;
    virtual DimensionType getType() const noexcept = 0;
    virtual std::size_t initialHash() const noexcept = 0;
    virtual std::pair<Dimension, Dimension> getInputs() const = 0;
    virtual std::pair<IteratorValue, IteratorValue> value(const IteratorValue& value) const = 0;
    ~MergeLikeOp() = default;

    Interface applyTo(const Interface& interface) const;
};

} // namespace kas
