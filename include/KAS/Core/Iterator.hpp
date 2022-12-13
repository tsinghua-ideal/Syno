#pragma once

#include <cstddef>
#include <sstream>
#include <variant>
#include <memory>
#include <map>
#include <vector>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/PrimitiveOp.hpp"


namespace kas {

using IteratorTransform = std::variant<
    std::unique_ptr<RepeatLikePrimitiveOp>,
    std::shared_ptr<SplitLikePrimitiveOp>,
    std::unique_ptr<MergeLikePrimitiveOp>,
    TensorStub
>;

class IteratorEvaluator;

class Iterator: public std::enable_shared_from_this<Iterator> {
protected:
    IteratorTransform parent;
    std::shared_ptr<Size> size;
public:
    Iterator(IteratorTransform parent, std::shared_ptr<Size> size);

    std::shared_ptr<Size> getSize() const;

    // Returns true on success
    bool compute(IteratorEvaluator& iteratorEvaluator);
};

class IteratorValueVisitor;
class BinaryOpValueNode;

struct IteratorValue: public std::enable_shared_from_this<IteratorValue> {
    virtual void accept(IteratorValueVisitor& visitor) = 0;
    std::shared_ptr<BinaryOpValueNode> operator+(IteratorValue& other);
    std::shared_ptr<BinaryOpValueNode> operator-(IteratorValue& other);
    std::shared_ptr<BinaryOpValueNode> operator*(IteratorValue& other);
    std::shared_ptr<BinaryOpValueNode> operator%(IteratorValue& other);
    std::shared_ptr<BinaryOpValueNode> operator/(IteratorValue& other);
    static std::vector<std::shared_ptr<IteratorValue>> DefaultAccessForShape(const Shape& shape, BindingContext& ctx);
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
    std::stringstream ss;
public:
    IteratorValuePrinter(const BindingContext& ctx);
    void visit(VariableValueNode& value) override;
    void visit(ConstValueNode& value) override;
    void visit(ImmediateValueNode& value) override;
    void visit(BinaryOpValueNode& value) override;
    std::string toString(IteratorValue& value);
};

} // namespace kas
