#pragma once

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Shape.hpp"


namespace kas {

class Iterator;

class CodeGenContext {
    friend class HalideGen;

public:
    struct TensorMetadata {
        std::string name;
        TensorMetadata() = default;
        TensorMetadata(std::string_view name);
    };

    struct IteratorVariableMetadata {
        std::string name;
        IteratorVariableMetadata() = default;
        IteratorVariableMetadata(std::string_view name);
    };

protected:
    std::vector<TensorMetadata> tensorMetadata;
    std::vector<std::pair<std::shared_ptr<Iterator>, IteratorVariableMetadata>> iteratorVariableMetadata;

    std::vector<std::size_t> outerLoopIterators;

public:
    std::string_view getTensorName(std::size_t index) const;
    std::size_t addTensor(std::string_view name);

    std::string_view getIteratorVariableName(std::size_t index) const;
    std::size_t addIteratorVariable(std::shared_ptr<Iterator> iterator, bool isOuterLoopIterator);

    // Returns the outer loop initializers and depth of the loops.
    std::pair<std::string, std::size_t> printOuterLoopsHeader(const BindingContext& ctx) const;
    std::string printOuterLoopsTail() const;
    std::string outerLoopIteratorsToString() const;
};

struct VariableValueNode;
struct ConstValueNode;
struct ImmediateValueNode;
struct BinaryOpValueNode;

class IteratorValueVisitor {
public:
    virtual void visit(VariableValueNode& value) = 0;
    virtual void visit(ConstValueNode& value) = 0;
    virtual void visit(ImmediateValueNode& value) = 0;
    virtual void visit(BinaryOpValueNode& value) = 0;
};

struct IteratorValueImpl {
    virtual void accept(IteratorValueVisitor& visitor) = 0;
};

struct IteratorValue {
    std::shared_ptr<IteratorValueImpl> value;
    explicit IteratorValue() = default;
    inline explicit IteratorValue(std::shared_ptr<IteratorValueImpl> value): value { std::move(value) } {}
    inline bool hasValue() const { return value != nullptr; }
    inline void accept(IteratorValueVisitor& visitor) const { value->accept(visitor); }
    IteratorValue operator+(const IteratorValue& other) const;
    IteratorValue operator-(const IteratorValue& other) const;
    IteratorValue operator*(const IteratorValue& other) const;
    IteratorValue operator%(const IteratorValue& other) const;
    IteratorValue operator/(const IteratorValue& other) const;
    static std::vector<IteratorValue> DefaultAccessForShape(const std::vector<std::shared_ptr<Iterator>>& interface, CodeGenContext& ctx);
};

struct VariableValueNode final: public IteratorValueImpl {
    std::size_t variableId;
    inline VariableValueNode(std::size_t variableId): variableId { variableId } {}
    inline void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static inline IteratorValue create(std::size_t variableId) { return IteratorValue(std::make_shared<VariableValueNode>(variableId)); }
};

struct ConstValueNode final: public IteratorValueImpl {
    std::shared_ptr<Size> value;
    inline ConstValueNode(std::shared_ptr<Size> value): value { std::move(value) } {}
    inline void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static inline IteratorValue create(std::shared_ptr<Size> value) { return IteratorValue(std::make_shared<ConstValueNode>(std::move(value))); }
};

struct ImmediateValueNode final: public IteratorValueImpl {
    int value;
    inline ImmediateValueNode(int value) : value { value } {}
    inline void accept(IteratorValueVisitor& visitor) override { visitor.visit(*this); }
    static inline IteratorValue create(int value) { return IteratorValue(std::make_shared<ImmediateValueNode>(value)); }
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
    static inline IteratorValue create(Type type, auto&& op1, auto&& op2) {
        return IteratorValue(std::make_shared<BinaryOpValueNode>(type, std::forward<decltype(op1)>(op1), std::forward<decltype(op2)>(op2)));
    }
};

class IteratorValuePrinter final: public IteratorValueVisitor {
    const BindingContext& ctx;
    const CodeGenContext& cgCtx;
    std::stringstream ss;
public:
    IteratorValuePrinter(const BindingContext& ctx, const CodeGenContext& cgCtx);
    void visit(VariableValueNode& value) override;
    void visit(ConstValueNode& value) override;
    void visit(ImmediateValueNode& value) override;
    void visit(BinaryOpValueNode& value) override;
    std::string toString(const IteratorValue& value);
};

struct ConditionalValue {
    enum class Type {
        Greater, Less, GreaterEq, LessEq
    };
    Type type;
    IteratorValue op1, op2;
    ConditionalValue(Type type, auto&& op1, auto&& op2):
        type { type },
        op1 { std::forward<decltype(op1)>(op1) },
        op2 { std::forward<decltype(op2)>(op2) }
    {}
};

} // namespace kas
