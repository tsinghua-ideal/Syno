#pragma once

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
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
};

struct IteratorValue {
    friend class HalideGen;
protected:
    std::shared_ptr<IteratorValueImpl> value;
public:
    explicit IteratorValue() = default;
    inline explicit IteratorValue(std::shared_ptr<IteratorValueImpl> value): value { std::move(value) } {}
    inline bool hasValue() const { return value != nullptr; }
    inline void accept(IteratorValueVisitor& visitor) const { value->accept(visitor); }
    IteratorValue operator+(const IteratorValue& other) const;
    IteratorValue operator-(const IteratorValue& other) const;
    IteratorValue operator*(const IteratorValue& other) const;
    IteratorValue operator%(const IteratorValue& other) const;
    IteratorValue operator/(const IteratorValue& other) const;
};

struct VariableValueNode final: public IteratorValueImpl {
    std::size_t variableId;
    inline VariableValueNode(std::size_t variableId): variableId { variableId } {}
    inline void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static inline IteratorValue Create(std::size_t variableId) { return IteratorValue(std::make_shared<VariableValueNode>(variableId)); }
};

struct ConstValueNode final: public IteratorValueImpl {
    std::shared_ptr<Size> value;
    inline ConstValueNode(std::shared_ptr<Size> value): value { std::move(value) } {}
    inline void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static inline IteratorValue Create(std::shared_ptr<Size> value) { return IteratorValue(std::make_shared<ConstValueNode>(std::move(value))); }
};

struct ImmediateValueNode final: public IteratorValueImpl {
    int value;
    inline ImmediateValueNode(int value) : value { value } {}
    inline void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static inline IteratorValue Create(int value) { return IteratorValue(std::make_shared<ImmediateValueNode>(value)); }
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
    inline void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static inline IteratorValue Create(Type type, auto&& op1, auto&& op2) {
        return IteratorValue(std::make_shared<BinaryOpValueNode>(type, std::forward<decltype(op1)>(op1), std::forward<decltype(op2)>(op2)));
    }
};

// When used as an IteratorValue, this is equivalent to a clamp(input, min, max - 1). When used to implement zero padding, this is equivalent to select(min <= input && input < max && other_clauses..., likely(input_tensor[access]), 0)
struct IntervalBoundValueNode final: public IteratorValueImpl {
    IteratorValue input;
    IteratorValue min, max;
    IntervalBoundValueNode(auto&& input, auto&& min, auto&& max):
        input { std::forward<decltype(input)>(input) },
        min { std::forward<decltype(min)>(min) },
        max { std::forward<decltype(max)>(max) }
    {}
    inline void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static inline IteratorValue Create(auto&& input, auto&& min, auto&& max) {
        return IteratorValue(std::make_shared<IntervalBoundValueNode>(std::forward<decltype(input)>(input), std::forward<decltype(min)>(min), std::forward<decltype(max)>(max)));
    }
};

class IteratorValuePrinter final: public IteratorValueVisitor {
    const BindingContext& ctx;
    std::stringstream ss;
public:
    IteratorValuePrinter(const BindingContext& ctx);
    void visit(VariableValueNode& value) override;
    void visit(ConstValueNode& value) override;
    void visit(ImmediateValueNode& value) override;
    void visit(BinaryOpValueNode& value) override;
    void visit(IntervalBoundValueNode& value) override;
    std::string toString(const IteratorValue& value);
};

} // namespace kas
