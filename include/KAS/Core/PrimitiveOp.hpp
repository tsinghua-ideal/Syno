#pragma once

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

// A dimension connects two Op's. So there is a certain direction when we are traversing the graph.
enum class Direction: bool {
    Down = false,
    Up = true,
};

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
        void accept(DimVisitor& visitor) const final override;
        inline const RepeatLikeOp *getOp() const noexcept { return op; }
    };
    Dimension output;
    RepeatLikeOp(auto&& output):
        output { std::forward<decltype(output)>(output) }
    {}
    RepeatLikeOp(const RepeatLikeOp&) = delete; // Do not copy! We want to store inputs in this class.
    RepeatLikeOp(RepeatLikeOp&&) = delete; // Do not move! Same reason.
    virtual DimensionType getType() const noexcept = 0;
    virtual std::size_t initialHash() const noexcept = 0;
    // We would like to store the DimensionImpl inside this class, so we can just return a reference to part of this object.
    virtual Dimension getInput() const = 0;

    template<typename Value>
    struct Values {
        Value input = {};
        Value output = {};
    };
    struct IteratorValues: public Values<IteratorValue> {
        bool known() const { return input && output; }
    };
    // Compute the iterators based on given iterators.
    // Only return the newly computed IteratorValue.
    virtual IteratorValues value(const IteratorValues& known) const = 0;
    // When evaluating dimensions, there are certain orderings of evaluations. For example, in a MergeOp, the iterators are i, j -> k. If we know j, then i must be evaluated before k.
    // Ordering is represented by integers. -1 means excluded from ordering. The difference between values represents the relative priority of the two dimensions. The greater the value, the earlier it should be evaluated.
    // Only return constraints for unevaluated dimensions.
    using OrderingValues = Values<int>;
    virtual OrderingValues ordering(const IteratorValues& known) const = 0;

    virtual inline std::pair<bool, CompactColorType> transformColor(CompactColorType fro) const { return { true, fro }; }
    virtual bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const = 0;

    ~RepeatLikeOp() = default;
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
        void accept(DimVisitor& visitor) const final override;
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

    template<typename Value>
    struct Values {
        Value input = {};
        Value outputLhs = {};
        Value outputRhs = {};
    };
    struct IteratorValues: public Values<IteratorValue> {
        bool known() const { return input && outputLhs && outputRhs; }
    };
    // Only return the newly computed IteratorValue.
    virtual IteratorValues value(const IteratorValues& known) const = 0;
    // Only return constraints for unevaluated dimensions.
    using OrderingValues = Values<int>;
    virtual OrderingValues ordering(const IteratorValues& known) const = 0;

    virtual inline std::tuple<bool, CompactColorType, CompactColorType> transformColor(CompactColorType fro) const { return { true, fro, fro }; }
    virtual bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const = 0;

    ~SplitLikeOp() = default;
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
        void accept(DimVisitor& visitor) const final override;
        inline const MergeLikeOp *getOp() const noexcept { return op; }
        inline Order getOrder() const noexcept { return order; }
        inline Dimension getOther() const noexcept {
            return order == Order::Left ? op->getInputR() : op->getInputL();
        }
    };
    Dimension output;
    MergeLikeOp(auto&& output):
        output { std::forward<decltype(output)>(output) }
    {}
    MergeLikeOp(const MergeLikeOp&) = delete;
    MergeLikeOp(MergeLikeOp&&) = delete;
    virtual DimensionType getType() const noexcept = 0;
    virtual std::size_t initialHash() const noexcept = 0;
    virtual Dimension getInputL() const = 0;
    virtual Dimension getInputR() const = 0;

    template<typename Value>
    struct Values {
        Value inputLhs = {};
        Value inputRhs = {};
        Value output = {};
    };
    struct IteratorValues: public Values<IteratorValue> {
        bool known() const { return inputLhs && inputRhs && output; }
    };
    // Only return the newly computed IteratorValue.
    virtual IteratorValues value(const IteratorValues& known) const = 0;
    // Only return constraints for unevaluated dimensions.
    using OrderingValues = Values<int>;
    virtual OrderingValues ordering(const IteratorValues& known) const = 0;

    virtual inline std::pair<bool, CompactColorType> transformColor(CompactColorType fro1, CompactColorType fro2) const { return { true, fro1 | fro2 }; }
    virtual bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const = 0;

    ~MergeLikeOp() = default;
};

template<typename Op>
concept PrimitiveOp = std::same_as<Op, RepeatLikeOp> || std::same_as<Op, SplitLikeOp> || std::same_as<Op, MergeLikeOp>;

} // namespace kas
