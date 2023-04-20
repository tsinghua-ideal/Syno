#pragma once

#include <array>
#include <variant>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Colors.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

// The value of a dimension.
class Valuation {
public:
    enum class Type: std::uint8_t {
        // Nothing can be derived about this dimension. Such a dimension can be free.
        Unoriented = 0,
        // We have derived that, at some future point, this dimension will be assigned due to value propagation, and we have determined the direction of this propagation. This means this dimension is not free.
        Oriented = 1,
        // This dimension has already been assigned with a value.
        Valued = 2,
    };
    using enum Type;

private:
    std::variant<std::monostate, Direction, IteratorValue> value;

public:
    inline Valuation() {}
    inline Valuation(Direction dir): value { dir } {}
    inline Valuation(const IteratorValue& val): value { val } {}

    inline Type type() const noexcept {
        return static_cast<Type>(value.index());
    }
    inline void assertCanBeConvertedFrom(const Valuation& other) const {
        KAS_ASSERT(type() >= other.type(), "Valuation of a dimension is decaying from {} to {}!", static_cast<int>(other.type()), static_cast<int>(type()));
        if (type() == Oriented && other.type() == Oriented) {
            auto dir = std::get<Direction>(value);
            auto otherDir = std::get<Direction>(other.value);
            KAS_ASSERT(dir == otherDir, "Valuation of a dimension is changing direction from {} to {}!", otherDir, dir);
        }
    }
    inline bool isRefined(const Valuation& other) const noexcept {
        return type() > other.type();
    }
    const IteratorValue& extract() const {
        KAS_ASSERT(type() == Valued, "Extracting a dimension which is not yet valued!");
        return std::get<IteratorValue>(value);
    }
};

template<std::size_t Count>
requires(Count <= 3)
struct Valuations {
    std::array<Valuation, Count> values;
    inline Valuation& operator[](std::uint8_t branch) noexcept { return values[static_cast<std::size_t>(branch)]; }
    inline const Valuation& operator[](std::uint8_t branch) const noexcept { return values[static_cast<std::size_t>(branch)]; }
    template<typename Fill>
    static Valuations<Count> FillBy(Fill&& fill) {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return Valuations<Count> { { fill(I)... } };
        }(std::make_index_sequence<Count>());
    }
};

class DimensionStore;

// There are 3 kinds of `PrimitiveOp`'s, listed below. Those classes can transform `Dimension`s, from those that index the output tensor, to forms that index the original tensors. So this is also kind of bottom-up.

// By repeat-like, we refer to the primitives that have one input iterator and one output iterator.
class RepeatLikeOp {
public:
    enum class Branch: std::uint8_t {
        Input = 0,
        Output = 1,
    };
    static constexpr std::uint8_t BranchCount = 2;

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

    using Values = Valuations<BranchCount>;
    // Compute the iterators based on given iterators. This also gives orientation, which is the direction of value propagation.
    virtual Values value(const Values& known) const = 0;

    virtual inline std::pair<bool, CompactColorType> transformColor(CompactColorType fro) const { return { true, fro }; }
    virtual bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const = 0;

    ~RepeatLikeOp() = default;
};

// By split-like, we refer to the primitives that have one input iterator and two output iterators.
class SplitLikeOp {
public:
    enum class Branch: std::int8_t {
        Input = 0,
        OutputLhs = 1,
        OutputRhs = 2,
    };
    static Branch OutputBranchFromOrder(Order order) noexcept {
        switch (order) {
        case Order::Left: return Branch::OutputLhs;
        case Order::Right: return Branch::OutputRhs;
        }
    }
    static constexpr std::uint8_t BranchCount = 3;

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

    using Values = Valuations<BranchCount>;
    virtual Values value(const Values& known) const = 0;

    virtual inline std::tuple<bool, CompactColorType, CompactColorType> transformColor(CompactColorType fro) const { return { true, fro, fro }; }
    virtual bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const = 0;

    ~SplitLikeOp() = default;
};

// By merge-like, we refer to the primitives that have two input iterators and one output iterator.
class MergeLikeOp {
public:
    enum class Branch: std::int8_t {
        InputLhs = 0,
        InputRhs = 1,
        Output = 2,
    };
    static Branch InputBranchFromOrder(Order order) noexcept {
        switch (order) {
        case Order::Left: return Branch::InputLhs;
        case Order::Right: return Branch::InputRhs;
        }
    }
    static constexpr std::uint8_t BranchCount = 3;

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

    using Values = Valuations<BranchCount>;
    virtual Values value(const Values& known) const = 0;

    virtual inline std::pair<bool, CompactColorType> transformColor(CompactColorType fro1, CompactColorType fro2) const { return { true, fro1 | fro2 }; }
    virtual bool transformInterface(ColoredInterface& interface, Colors& colors, Colors::Options options) const = 0;

    ~MergeLikeOp() = default;
};

template<typename Op>
concept PrimitiveOp = std::same_as<Op, RepeatLikeOp> || std::same_as<Op, SplitLikeOp> || std::same_as<Op, MergeLikeOp>;

} // namespace kas
