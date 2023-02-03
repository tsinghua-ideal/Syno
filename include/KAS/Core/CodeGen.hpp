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

class IteratorValueVisitor;
struct BinaryOpValueNode;

struct IteratorValue: public std::enable_shared_from_this<IteratorValue> {
    virtual void accept(IteratorValueVisitor& visitor) = 0;
    std::shared_ptr<BinaryOpValueNode> operator+(IteratorValue& other);
    std::shared_ptr<BinaryOpValueNode> operator-(IteratorValue& other);
    std::shared_ptr<BinaryOpValueNode> operator*(IteratorValue& other);
    std::shared_ptr<BinaryOpValueNode> operator%(IteratorValue& other);
    std::shared_ptr<BinaryOpValueNode> operator/(IteratorValue& other);
    static std::vector<std::shared_ptr<IteratorValue>> DefaultAccessForShape(const std::vector<std::shared_ptr<Iterator>>& interface, CodeGenContext& ctx);
    template<typename Derived>
    std::shared_ptr<Derived> shared_from_base() {
        return std::dynamic_pointer_cast<Derived>(shared_from_this());
    }
};

struct VariableValueNode: public IteratorValue {
    std::size_t variableId;
    VariableValueNode(std::size_t variableId);
    void accept(IteratorValueVisitor& visitor) override;
};

struct ConstValueNode: public IteratorValue {
    std::shared_ptr<Size> value;
    ConstValueNode(std::shared_ptr<Size> value);
    void accept(IteratorValueVisitor& visitor) override;
};

struct ImmediateValueNode: public IteratorValue {
    int value;
    ImmediateValueNode(int value);
    void accept(IteratorValueVisitor& visitor) override;
};

struct BinaryOpValueNode: public IteratorValue {
    enum class Type {
        Add, Sub, Mul, Mod, Div
    };
    Type type;
    std::shared_ptr<IteratorValue> op1, op2;
    BinaryOpValueNode(Type type, std::shared_ptr<IteratorValue> op1, std::shared_ptr<IteratorValue> op2);
    void accept(IteratorValueVisitor& visitor) override;
};

class IteratorValueVisitor {
public:
    virtual void visit(VariableValueNode& value) = 0;
    virtual void visit(ConstValueNode& value) = 0;
    virtual void visit(ImmediateValueNode& value) = 0;
    virtual void visit(BinaryOpValueNode& value) = 0;
};

class IteratorValuePrinter: public IteratorValueVisitor {
    const BindingContext& ctx;
    const CodeGenContext& cgCtx;
    std::stringstream ss;
public:
    IteratorValuePrinter(const BindingContext& ctx, const CodeGenContext& cgCtx);
    void visit(VariableValueNode& value) override;
    void visit(ConstValueNode& value) override;
    void visit(ImmediateValueNode& value) override;
    void visit(BinaryOpValueNode& value) override;
    std::string toString(IteratorValue& value);
};

} // namespace kas
