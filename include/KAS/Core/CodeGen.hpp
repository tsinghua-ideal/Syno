#pragma once

#include <compare>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Size.hpp"


namespace kas {

struct VariableValueNode;
struct ConstValueNode;
struct ImmediateValueNode;
struct BinaryOpValueNode;
struct IntervalBoundValueNode;

class IteratorValueVisitor {
public:
    virtual void visit(VariableValueNode& value) = 0;
    virtual void visit(ConstValueNode& value) = 0;
    virtual void visit(ImmediateValueNode& value) = 0;
    virtual void visit(BinaryOpValueNode& value) = 0;
    virtual void visit(IntervalBoundValueNode& value) = 0;
};

struct IteratorValueImpl {
    virtual void accept(IteratorValueVisitor& visitor) = 0;
    virtual ~IteratorValueImpl() = default;
};

struct IteratorValue {
    friend class HalideGen;
protected:
    std::shared_ptr<IteratorValueImpl> value;
public:
    IteratorValue() = default;
    IteratorValue(std::shared_ptr<IteratorValueImpl> value): value { std::move(value) } {}
    const std::shared_ptr<IteratorValueImpl>& get() const { return value; }
    bool hasValue() const { return value != nullptr; }
    explicit operator bool() const { return hasValue(); }
    void accept(IteratorValueVisitor& visitor) const { value->accept(visitor); }
    IteratorValue operator+(const IteratorValue& other) const;
    IteratorValue operator-(const IteratorValue& other) const;
    IteratorValue operator*(const IteratorValue& other) const;
    IteratorValue operator/(const IteratorValue& other) const;
    IteratorValue operator%(const IteratorValue& other) const;
    bool operator==(const IteratorValue& other) const = default;
    std::strong_ordering operator<=>(const IteratorValue& other) const = default;
    template<typename T>
    requires std::is_base_of_v<IteratorValueImpl, T>
    std::shared_ptr<T> tryAs() const { return std::dynamic_pointer_cast<T>(value); }
    template<typename T>
    requires std::is_base_of_v<IteratorValueImpl, T>
    T& as() const { return *tryAs<T>(); }
    std::string toString(const BindingContext& ctx) const;

    // FOR DEBUG USAGE ONLY!
    std::string debugToString() const {
        if (BindingContext::DebugPublicCtx) {
            return toString(*BindingContext::DebugPublicCtx);
        } else {
            return "NO_PUBLIC_CONTEXT";
        }
    }
};

struct VariableValueNode final: public IteratorValueImpl {
    bool isReduce;
    std::size_t index;
    std::string name;
    VariableValueNode(bool isReduce, std::size_t index, auto&& name): isReduce { isReduce }, index { index }, name { std::forward<decltype(name)>(name) } {}
    void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static IteratorValue Create(bool isReduce, std::size_t index, auto&& name) { return IteratorValue(std::make_shared<VariableValueNode>(isReduce, index, std::forward<decltype(name)>(name))); }
};

struct ConstValueNode final: public IteratorValueImpl {
    Size value;
    ConstValueNode(auto&& value): value { std::forward<decltype(value)>(value) } {}
    void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static IteratorValue Create(auto&& value) { return IteratorValue(std::make_shared<ConstValueNode>(std::forward<decltype(value)>(value))); }
};

struct ImmediateValueNode final: public IteratorValueImpl {
    int value;
    ImmediateValueNode(int value) : value { value } {}
    void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static IteratorValue Create(int value) { return IteratorValue(std::make_shared<ImmediateValueNode>(value)); }
    static const IteratorValue Zero;
    static const IteratorValue One;
    static const IteratorValue Two;
};

struct BinaryOpValueNode final: public IteratorValueImpl {
    enum class Type {
        Add, Sub, Mul, Mod, Div
    };
    Type type;
    IteratorValue op1, op2;
    BinaryOpValueNode(Type type, auto&& op1, auto&& op2):
        type { type },
        op1 { std::forward<decltype(op1)>(op1) },
        op2 { std::forward<decltype(op2)>(op2) }
    {}
    void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static IteratorValue Create(Type type, auto&& op1, auto&& op2) {
        return IteratorValue(std::make_shared<BinaryOpValueNode>(type, std::forward<decltype(op1)>(op1), std::forward<decltype(op2)>(op2)));
    }
};

// When used as an IteratorValue, this is equivalent to a clamp(input, min, max - 1). When used to implement zero padding, this is equivalent to select(min <= input && input < max && other_clauses..., likely(input_tensor[access]), 0)
struct IntervalBoundValueNode final: public IteratorValueImpl {
    IteratorValue input;
    // min is 0.
    Size max;
    Size outOfBoundFraction;
    IntervalBoundValueNode(auto&& input, auto&& max, auto&& outOfBoundFraction):
        input { std::forward<decltype(input)>(input) },
        max { std::forward<decltype(max)>(max) },
        outOfBoundFraction { std::forward<decltype(outOfBoundFraction)>(outOfBoundFraction) }
    {}
    void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static IteratorValue Create(auto&& input, auto&& max, auto&& outOfBoundFraction) {
        return IteratorValue(std::make_shared<IntervalBoundValueNode>(std::forward<decltype(input)>(input), std::forward<decltype(max)>(max), std::forward<decltype(outOfBoundFraction)>(outOfBoundFraction)));
    }
};

class IteratorValuePrinter final: public IteratorValueVisitor {
    const BindingContext& ctx;
    std::stringstream ss;
public:
    IteratorValuePrinter(const BindingContext& ctx): ctx { ctx } {}
    void visit(VariableValueNode& value) override;
    void visit(ConstValueNode& value) override;
    void visit(ImmediateValueNode& value) override;
    void visit(BinaryOpValueNode& value) override;
    void visit(IntervalBoundValueNode& value) override;
    std::string toString(const IteratorValue& value);
};

} // namespace kas

template<>
struct std::hash<kas::IteratorValue> {
    std::size_t operator()(const kas::IteratorValue& value) const {
        return std::hash<std::shared_ptr<kas::IteratorValueImpl>>()(value.get());
    }
};
