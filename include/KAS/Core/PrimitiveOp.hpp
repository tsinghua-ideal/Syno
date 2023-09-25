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

private:
    std::variant<std::monostate, Direction, IteratorValue> value;

public:
    Valuation() {}
    Valuation(std::monostate) {}
    Valuation(Direction dir): value { dir } {}
    Valuation(const IteratorValue& val): value { val } {}

    Type type() const noexcept {
        return static_cast<Type>(value.index());
    }
    bool isUnoriented() const noexcept { return type() == Type::Unoriented; }
    bool isOriented() const noexcept { return type() == Type::Oriented; }
    bool isOrientedUp() const { return isOriented() && std::get<Direction>(value) == Direction::Up; }
    bool isOrientedDown() const { return isOriented() && std::get<Direction>(value) == Direction::Down; }
    bool isUnorientedOrOrientedUp() const { return isUnoriented() || isOrientedUp(); }
    bool isUnorientedOrOrientedDown() const { return isUnoriented() || isOrientedDown(); }
    bool isValued() const noexcept { return type() == Type::Valued; }
    bool isValuedOrOrientedUp() const { return isValued() || isOrientedUp(); }
    bool isValuedOrOrientedDown() const { return isValued() || isOrientedDown(); }
    void assertCanBeConvertedFrom(const Valuation& other) const {
        KAS_ASSERT(type() >= other.type(), "Valuation of a dimension is decaying from {} to {}!", static_cast<int>(other.type()), static_cast<int>(type()));
        if (type() == Type::Oriented && other.type() == Type::Oriented) {
            auto dir = std::get<Direction>(value);
            auto otherDir = std::get<Direction>(other.value);
            KAS_ASSERT(dir == otherDir, "Valuation of a dimension is changing direction from {} to {}!", otherDir, dir);
        }
    }
    bool isRefined(const Valuation& other) const noexcept {
        return type() > other.type();
    }
    Direction extractOrientation() const {
        KAS_ASSERT(type() == Type::Oriented, "Extracting a direction from a dimension which is not oriented!");
        return std::get<Direction>(value);
    }
    std::optional<Direction> tryOrientation() const {
        if (type() == Type::Oriented) {
            return std::get<Direction>(value);
        } else {
            return std::nullopt;
        }
    }
    const IteratorValue& extractValue() const {
        KAS_ASSERT(type() == Type::Valued, "Extracting a dimension which is not yet valued!");
        return std::get<IteratorValue>(value);
    }
    IteratorValue tryValue() const {
        if (type() == Type::Valued) {
            return std::get<IteratorValue>(value);
        } else {
            return {};
        }
    }
    // Pattern match.
    template<typename CaseUnoriented, typename CaseOriented, typename CaseValued, typename Result = std::invoke_result_t<CaseUnoriented>>
    requires(
        std::invocable<CaseUnoriented> &&
        std::invocable<CaseOriented, Direction> &&
        std::invocable<CaseValued, const IteratorValue&> &&
        std::same_as<Result, std::invoke_result_t<CaseUnoriented>> &&
        std::same_as<Result, std::invoke_result_t<CaseOriented, Direction>> &&
        std::same_as<Result, std::invoke_result_t<CaseValued, const IteratorValue&>>
    )
    Result match(CaseUnoriented&& caseUnoriented, CaseOriented&& caseOriented, CaseValued&& caseValued) const {
        if constexpr (std::is_void_v<Result>) {
            switch (type()) {
                case Type::Unoriented: caseUnoriented(); break;
                case Type::Oriented: caseOriented(std::get<Direction>(value)); break;
                case Type::Valued: caseValued(std::get<IteratorValue>(value)); break;
            }
        } else {
            switch (type()) {
                case Type::Unoriented: return caseUnoriented();
                case Type::Oriented: return caseOriented(std::get<Direction>(value));
                case Type::Valued: return caseValued(std::get<IteratorValue>(value));
            }
        }
    }
};

template<std::size_t Count>
requires(Count <= 3)
struct Valuations {
    std::array<Valuation, Count> values;
    // All Dimension's are valued.
    bool allValued() const noexcept {
        return std::all_of(values.begin(), values.end(), [](const Valuation& val) { return val.isValued(); });
    }
    // We know nothing about these Dimension's.
    bool allUnoriented() const noexcept {
        return std::all_of(values.begin(), values.end(), [](const Valuation& val) { return val.isUnoriented(); });
    }
    // Nothing to deduce if and only if all known or all unknown.
    bool canSkipDeduction() const noexcept {
        return allValued() || allUnoriented();
    }
    Valuation& operator[](std::uint8_t branch) noexcept { return values[static_cast<std::size_t>(branch)]; }
    const Valuation& operator[](std::uint8_t branch) const noexcept { return values[static_cast<std::size_t>(branch)]; }
    template<typename Fill>
    static Valuations<Count> FillBy(Fill&& fill) {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return Valuations<Count> { { fill(I)... } };
        }(std::make_index_sequence<Count>());
    }
};

class ExpandOp;
class ReduceOp;
class MergeOp;
class ShareOp;
class ShiftOp;
class SplitOp;
class StrideOp;
class UnfoldOp;
class OpVisitor {
public:
    virtual void visit(const ExpandOp& op) = 0;
    virtual void visit(const ReduceOp& op) = 0;
    virtual void visit(const MergeOp& op) = 0;
    virtual void visit(const ShareOp& op) = 0;
    virtual void visit(const ShiftOp& op) = 0;
    virtual void visit(const SplitOp& op) = 0;
    virtual void visit(const StrideOp& op) = 0;
    virtual void visit(const UnfoldOp& op) = 0;
};

class PrimitiveOpStore;

// There are 3 kinds of `PrimitiveOp`'s, listed below. Those classes can transform `Dimension`s, from those that index the output tensor, to forms that index the original tensors. So this is also kind of bottom-up.
// First we define a common base class.
class PrimitiveOp {
public:
    Color color;
    PrimitiveOp(auto&& color):
        color { std::forward<decltype(color)>(color) } {}
    virtual DimensionType getType() const noexcept = 0;
    virtual std::size_t initialHash() const noexcept = 0;
    virtual std::size_t opHash() const noexcept = 0;
    virtual void accept(OpVisitor& visitor) const = 0;
    virtual bool canApplyToInterface(const GraphHandle& interface) const = 0;
    virtual GraphHandle applyToInterface(const GraphHandle& interface) const = 0;
    virtual std::string description(const BindingContext& ctx) const = 0;
    virtual std::string descendantsDescription(const BindingContext& ctx) const = 0;
    virtual ~PrimitiveOp() = default;
};

template<typename Op>
concept PrimitiveOpImpl = std::derived_from<Op, PrimitiveOp> && requires {
    { Op::Type } -> std::convertible_to<DimensionType>;
};

#define KAS_REPORT_OP_HASH_COLLISION(op1, op2) do { \
    KAS_WARNING("Duplicate PrimitiveOp's! Or even worse, hash collision. {}", \
        ::kas::BindingContext::DebugPublicCtx != nullptr ? ::fmt::format("Maybe helpful: {} vs {}.", \
            (op1).descendantsDescription(*::kas::BindingContext::DebugPublicCtx), \
            (op2).descendantsDescription(*::kas::BindingContext::DebugPublicCtx)) \
        : "Please call Sampler._bind_debug_context() for more information."); \
} while (false)

// By repeat-like, we refer to the primitives that have one input iterator and one output iterator.
class RepeatLikeOp: public PrimitiveOp {
public:
    enum class Branch: std::uint8_t {
        Input = 0,
        Output = 1,
    };
    static constexpr std::uint8_t BranchCount = 2;

    class Input: public DimensionImpl {
    protected:
        const RepeatLikeOp *op;
        Input(const RepeatLikeOp *op): op { op } {}
    public:
        template<typename Derived>
        const Derived *getDerivedOp() const noexcept {
            return static_cast<const Derived *>(op);
        }
        std::size_t hash() const noexcept final override {
            return op->opHash();
        }
        void accept(DimVisitor& visitor) const final override;
        const Color& getColor() const final override { return op->color; }
        const RepeatLikeOp *getOp() const noexcept { return op; }
    };
    Dimension output;
    RepeatLikeOp(const Dimension& output):
        PrimitiveOp { output.getColor() },
        output { output }
    {}
    RepeatLikeOp(const RepeatLikeOp&) = delete; // Do not copy! We want to store inputs in this class.
    RepeatLikeOp(RepeatLikeOp&&) = delete; // Do not move! Same reason.
    std::size_t opHash() const noexcept final override {
        std::size_t h = initialHash();
        HashCombineRaw(h, output.hash());
        return h;
    }
    // We would like to store the DimensionImpl inside this class, so we can just return a reference to part of this object.
    virtual Dimension getInput() const = 0;

    using Values = Valuations<BranchCount>;
    // Compute the iterators based on given iterators. This also gives orientation, which is the direction of value propagation.
    virtual Values value(const Values& known) const = 0;

    virtual std::pair<bool, CompactColor> transformColor(CompactColor fro) const { return { true, fro }; }
    bool canApplyToInterface(const GraphHandle& interface) const final override {
        return interface.contains(output);
    }
    GraphHandle applyToInterface(const GraphHandle& interface) const final override {
        return interface.substitute1to1(output, getInput());
    }

    std::string description(const BindingContext& ctx) const final override {
        return fmt::format("{} -> {}", getInput().description(ctx), output.description(ctx));
    }
    std::string descendantsDescription(const BindingContext& ctx) const final override {
        return fmt::format("{} -> {}", getInput().description(ctx), output.descendantsDescription(ctx));
    }
};

// By split-like, we refer to the primitives that have one input iterator and two output iterators.
class SplitLikeOp: public PrimitiveOp {
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
    static Branch OtherOutputBranch(Branch branch) {
        KAS_ASSERT(branch == Branch::OutputLhs || branch == Branch::OutputRhs);
        if (branch == Branch::OutputLhs) {
            return Branch::OutputRhs;
        } else {
            return Branch::OutputLhs;
        }
    }
    static constexpr std::uint8_t BranchCount = 3;

    class Input: public DimensionImpl {
    protected:
        const SplitLikeOp *op;
        Input(const SplitLikeOp *op): op { op } {}
    public:
        template<typename Derived>
        const Derived *getDerivedOp() const noexcept {
            return static_cast<const Derived *>(op);
        }
        std::size_t hash() const noexcept final override {
            return op->opHash();
        }
        void accept(DimVisitor& visitor) const final override;
        const Color& getColor() const final override { return op->color; }
        const SplitLikeOp *getOp() const noexcept { return op; }
    };
    Dimension outputLhs, outputRhs;
    SplitLikeOp(const Dimension& outputLhs, const Dimension& outputRhs):
        PrimitiveOp { Color { outputLhs.getColor(), outputRhs.getColor() } },
        outputLhs { outputLhs },
        outputRhs { outputRhs }
    {}
    SplitLikeOp(const SplitLikeOp&) = delete;
    SplitLikeOp(SplitLikeOp&&) = delete;
    std::size_t opHash() const noexcept final override {
        std::size_t h = initialHash();
        HashCombineRaw(h, outputLhs.hash());
        constexpr int SizeTypeWidth = std::numeric_limits<std::size_t>::digits;
        HashCombineRaw(h, std::rotl(outputRhs.hash(), SizeTypeWidth / 2));
        return h;
    }
    virtual Dimension getInput() const = 0;

    using Values = Valuations<BranchCount>;
    virtual Values value(const Values& known) const = 0;

    virtual std::tuple<bool, CompactColor, CompactColor> transformColor(CompactColor fro) const { return { true, fro, fro }; }
    bool canApplyToInterface(const GraphHandle& interface) const final override {
        return interface.contains(outputLhs) && interface.contains(outputRhs);
    }
    GraphHandle applyToInterface(const GraphHandle& interface) const final override {
        return interface.substitute2to1(outputLhs, outputRhs, getInput());
    }

    std::string description(const BindingContext& ctx) const final override {
        return fmt::format("{} -> {}, {}", getInput().description(ctx), outputLhs.description(ctx), outputRhs.description(ctx));
    }
    std::string descendantsDescription(const BindingContext& ctx) const final override {
        return fmt::format("{} -> {}, {}", getInput().description(ctx), outputLhs.descendantsDescription(ctx), outputRhs.descendantsDescription(ctx));
    }
};

// By merge-like, we refer to the primitives that have two input iterators and one output iterator.
class MergeLikeOp: public PrimitiveOp {
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
        Input(const MergeLikeOp *op, Order order): op { op }, order { order } {}
    public:
        template<typename Derived>
        const Derived *getDerivedOp() const noexcept {
            return static_cast<const Derived *>(op);
        }
        std::size_t hash() const noexcept final override {
            std::size_t h = op->opHash();
            HashCombine(h, order == Order::Left ? 0 : std::numeric_limits<std::size_t>::max());
            return h;
        }
        virtual bool is(DimensionTypeWithOrder ty) const noexcept override = 0;
        void accept(DimVisitor& visitor) const final override;
        const Color& getColor() const final override { return op->color; }
        const MergeLikeOp *getOp() const noexcept { return op; }
        Order getOrder() const noexcept { return order; }
        Dimension getOther() const noexcept {
            return order == Order::Left ? op->getInputR() : op->getInputL();
        }
    };
    Dimension output;
    MergeLikeOp(const Dimension& output):
        PrimitiveOp { Color { output.getColor() } },
        output { output }
    {}
    MergeLikeOp(const MergeLikeOp&) = delete;
    MergeLikeOp(MergeLikeOp&&) = delete;
    std::size_t opHash() const noexcept final override {
        std::size_t h = initialHash();
        HashCombineRaw(h, output.hash());
        return h;
    }
    virtual Dimension getInputL() const = 0;
    virtual Dimension getInputR() const = 0;

    using Values = Valuations<BranchCount>;
    virtual Values value(const Values& known) const = 0;

    virtual std::pair<bool, CompactColor> transformColor(CompactColor fro1, CompactColor fro2) const { return { true, fro1 | fro2 }; }
    bool canApplyToInterface(const GraphHandle& interface) const final override {
        return interface.contains(output);
    }
    GraphHandle applyToInterface(const GraphHandle& interface) const final override {
        return interface.substitute1to2(output, getInputL(), getInputR());
    }

    std::string description(const BindingContext& ctx) const final override {
        return fmt::format("{}, {} -> {}", getInputL().description(ctx), getInputR().description(ctx), output.description(ctx));
    }
    std::string descendantsDescription(const BindingContext& ctx) const final override {
        return fmt::format("{}, {} -> {}", getInputL().description(ctx), getInputR().description(ctx), output.descendantsDescription(ctx));
    }
};

} // namespace kas

template<>
struct fmt::formatter<kas::Valuation::Type>: formatter<string_view> {
    template<typename FormatContext>
    auto format(kas::Valuation::Type v, FormatContext& ctx) const {
        string_view name = "Unknown";
        switch (v) {
        using namespace std::string_view_literals;
        case kas::Valuation::Type::Unoriented: name = "Unoriented"sv; break;
        case kas::Valuation::Type::Oriented: name = "Oriented"sv; break;
        case kas::Valuation::Type::Valued: name = "Valued"sv; break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};

template<>
struct fmt::formatter<kas::Valuation>: formatter<string_view> {
    template<typename FormatContext>
    auto format(kas::Valuation v, FormatContext& ctx) const {
        string_view name = "Unknown";
        switch (v.type()) {
        using namespace std::string_view_literals;
        case kas::Valuation::Type::Unoriented: name = "Unoriented"sv; break;
        case kas::Valuation::Type::Oriented:
            switch (v.extractOrientation()) {
            case kas::Direction::Up: name = "Oriented Up"sv; break;
            case kas::Direction::Down: name = "Oriented Down"sv; break;
            }
            break;
        case kas::Valuation::Type::Valued: name = "Valued"sv; break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
