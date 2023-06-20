#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Core/MapReduce.hpp"
#include "KAS/Core/Shape.hpp"


namespace kas {

class PureTensor {
protected:
    std::string name;
    std::vector<Dimension> dims;
public:
    PureTensor(auto&& name, auto&& dims):
        name { std::forward<decltype(name)>(name) },
        dims { std::forward<decltype(dims)>(dims) }
    {}
    bool operator==(const PureTensor& rhs) const {
        return name == rhs.name && dims == rhs.dims;
    }
    const std::string& getName() const { return name; }
    const std::vector<Dimension>& getDimensions() const { return dims; }
    ShapeView getShape() const { return ShapeView(dims); }
    std::string shapeToString(const BindingContext& ctx) const;
    std::string description(const BindingContext& ctx) const;
};

class IntegerTensorExpression;
class TensorTensorExpression;
class BinaryOpTensorExpression;

class TensorExpressionVisitor {
public:
    virtual void visit(IntegerTensorExpression& expr) = 0;
    virtual void visit(TensorTensorExpression& expr) = 0;
    virtual void visit(BinaryOpTensorExpression& expr) = 0;
    virtual ~TensorExpressionVisitor() = default;
};

template<typename Derived, typename T>
class ValuedTensorExpressionVisitor: public TensorExpressionVisitor {
    T storedResult;
public:
    void visit(IntegerTensorExpression& expr) override {
        storedResult = static_cast<Derived*>(this)->visits(expr);
    }
    void visit(TensorTensorExpression& expr) override {
        storedResult = static_cast<Derived*>(this)->visits(expr);
    }
    void visit(BinaryOpTensorExpression& expr) override {
        storedResult = static_cast<Derived*>(this)->visits(expr);
    }
    T result() { return std::move(storedResult); }
};

class TensorExpressionImpl {
public:
    virtual void accept(TensorExpressionVisitor& visitor) = 0;
    virtual std::string toString() const = 0;
    virtual ~TensorExpressionImpl() = default;
};

class TensorExpression {
    std::shared_ptr<TensorExpressionImpl> value;
public:
    using Position = int;
    constexpr static Position Output = -1;
    template<int Index>
    constexpr static Position Input = Index;
    // -1 for output tensor, otherwise index of input tensors.

    TensorExpression() = default;
    explicit TensorExpression(std::shared_ptr<TensorExpressionImpl> value): value { std::move(value) } {}

    explicit operator bool() const { return static_cast<bool>(value); }

    template<std::derived_from<TensorExpressionImpl> T>
    std::shared_ptr<T> tryAs() const {
        return std::dynamic_pointer_cast<T>(value);
    }
    template<std::derived_from<TensorExpressionImpl> T>
    bool is() const { return static_cast<bool>(tryAs<T>()); }

    TensorExpression operator+(const TensorExpression& other) const;
    TensorExpression& operator+=(const TensorExpression& other);
    TensorExpression operator*(const TensorExpression& other) const;
    TensorExpression& operator*=(const TensorExpression& other);

    void accept(TensorExpressionVisitor& visitor) const { value->accept(visitor); }
    std::string toString() const { return value->toString(); }
};

class IntegerTensorExpression final: public TensorExpressionImpl {
public:
    int value;
    IntegerTensorExpression(int value): value { value } {}
    void accept(TensorExpressionVisitor& visitor) override { visitor.visit(*this); }
    std::string toString() const override { return std::to_string(value); }

    static TensorExpression Create(int value);
};

class TensorTensorExpression final: public TensorExpressionImpl {
public:
    TensorExpression::Position position;
    TensorTensorExpression(TensorExpression::Position position): position { position } {}
    void accept(TensorExpressionVisitor& visitor) override { visitor.visit(*this); }
    std::string toString() const override;

    static TensorExpression Create(TensorExpression::Position position) {
        return TensorExpression(std::make_shared<TensorTensorExpression>(position));
    }
};

class BinaryOpTensorExpression final: public TensorExpressionImpl {
public:
    enum class Op {
        Add, Mul,
    };
    TensorExpression lhs;
    TensorExpression rhs;
    Op op;
    BinaryOpTensorExpression(TensorExpression lhs, TensorExpression rhs, Op op):
        lhs { std::move(lhs) },
        rhs { std::move(rhs) },
        op { op }
    {}
    void accept(TensorExpressionVisitor& visitor) override { visitor.visit(*this); }
    std::string toString() const override;

    // Perform some simple canonicalization.
    static TensorExpression Create(TensorExpression lhs, TensorExpression rhs, Op op);
};

struct AbstractAccess {
    TensorExpression::Position position;
    std::vector<IteratorValue> outerLoops;
    std::vector<IteratorValue> innerLoops;
    Shape innerLoopsShape;
    std::vector<std::vector<IteratorValue>> inputs;
    std::vector<IteratorValue> output;

    // Description of the expression.
    TensorExpression expression; // The actual expression.
    std::optional<Size> divBy; // The divisor, if any.

    // The standard interface, i.e., the outer loops.
    std::string outerLoopsIteratorsToString() const;

    // The inner reduction loops.
    std::string innerLoopsIteratorsToString() const;

    // The access of the tensor at `pos`.
    std::string accessToString(const BindingContext& ctx, int pos) const;

    // The innermost statement.
    std::string statementToString(const BindingContext& ctx) const;

    // The result tensor.
    std::string targetEntryToString() const;
};

class TensorView {
protected:
    // The interface iterators.
    std::vector<const Iterator *> interface;
    // Define the corresponding shape view for interface.
    using IteratorShapeView = AbstractShape<const std::vector<const Iterator *>&, [](const Iterator * const& ptr) -> const Size& { return ptr->size(); }>;
    // The map-reduce iterators.
    std::vector<const MapReduce *> manipulations;
    using ReduceIteratorShapeView = AbstractShape<const std::vector<const MapReduce *>&, [](const MapReduce * const& ptr) -> const Size& { return ptr->size(); }>;

    // How to blend the tensors? TODO
    std::vector<PureTensor> tensors;
    AbstractAccess forwardAccess; // Iterators evaluated for the forward pipeline.
    std::vector<AbstractAccess> backwardAccesses; // Iterators evaluated for the backward pipeline.

public:
    // Build the tensor from iterator DAG.
    explicit TensorView(const std::vector<std::vector<Dimension>>& tensors);
    explicit TensorView(std::initializer_list<std::vector<Dimension>> tensors):
        TensorView { std::vector(tensors) } {}

    IteratorShapeView getInterfaceShape() const { return IteratorShapeView(interface); }

    // Returns the underlying tensor underneath the view.
    const std::vector<PureTensor>& getUnderlyingTensors() const { return tensors; }
    auto getUnderlyingTensorRange() const { return tensors | std::views::transform(&PureTensor::getDimensions); }
    // Returns all dimensions in the underlying tensors.
    auto getUnderlyingDimensions() const {
        return tensors
            | std::views::transform(&PureTensor::getDimensions)
            | std::views::join;
    }

    bool operator==(const TensorView& rhs) const {
        return std::ranges::equal(tensors, rhs.tensors);
    }
    std::size_t hash() const {
        return std::hash<std::vector<Interface>>{}(getUnderlyingTensorRange());
    }

    const AbstractAccess& getForwardAccess() const { return forwardAccess; }

    const std::vector<AbstractAccess>& getBackwardAccesses() const { return backwardAccesses; }

    // Returns the interface of the view.
    const std::vector<const Iterator *>& getInterfaceIterators() const { return interface; }

    // Returns the map-reduce manipulations.
    const std::vector<const MapReduce *>& getManipulations() const { return manipulations; }

    // Note that sometimes we need padding to make things work. Here we need to guarantee that all dimensions in the Graph are valid, so instead of padding tensors, we pad variables.
    ConcreteConsts computePadding(const BindingContext& ctx, const ConcreteConsts& consts) const;

    // Observe that FLOPs is determined by outer loops and inner loops.
    std::size_t getFLOPs(const ConcreteConsts& consts) const;

    // Evaluate the full loops.
    std::string printNestedLoops(const BindingContext& ctx, int pos) const;
    std::string printNestedLoopsForAll(const BindingContext& ctx) const;

    std::string description(const BindingContext& ctx) const;
};

} // namespace kas
