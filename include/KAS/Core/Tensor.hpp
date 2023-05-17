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
    inline const std::string& getName() const { return name; }
    inline const std::vector<Dimension>& getDimensions() const { return dims; }
    inline ShapeView getShape() const { return ShapeView(dims); }
    std::string shapeToString(const BindingContext& ctx) const;
    std::string description(const BindingContext& ctx) const;
};

struct AbstractAccess {
    constexpr static int Output = -1;
    template<int Index>
    constexpr static int Input = Index;
    // -1 for output tensor, otherwise index of input tensors.
    int position;
    std::vector<IteratorValue> outerLoops;
    std::vector<IteratorValue> innerLoops;
    Shape innerLoopsShape;
    std::vector<std::vector<IteratorValue>> inputs;
    std::vector<IteratorValue> output;

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
    std::vector<const MapReduceOp *> manipulations;
    using ReduceIteratorShapeView = AbstractShape<const std::vector<const MapReduceOp *>&, [](const MapReduceOp * const& ptr) -> const Size& { return ptr->size(); }>;

    // How to blend the tensors? TODO
    std::vector<PureTensor> tensors;
    AbstractAccess forwardAccess; // Iterators evaluated for the forward pipeline.
    std::vector<AbstractAccess> backwardAccesses; // Iterators evaluated for the backward pipeline.

public:
    // Build the tensor from iterator DAG.
    explicit TensorView(const std::vector<std::vector<Dimension>>& tensors);
    explicit inline TensorView(std::initializer_list<std::vector<Dimension>> tensors):
        TensorView { std::vector(tensors) } {}

    inline IteratorShapeView getInterfaceShape() const { return IteratorShapeView(interface); }

    // Returns the underlying tensor underneath the view.
    inline const std::vector<PureTensor>& getUnderlyingTensors() const { return tensors; }
    // Returns all dimensions in the underlying tensors.
    inline auto getUnderlyingDimensions() const {
        return tensors
            | std::views::transform(&PureTensor::getDimensions)
            | std::views::join;
    }

    inline const AbstractAccess& getForwardAccess() const { return forwardAccess; }

    inline const std::vector<AbstractAccess>& getBackwardAccesses() const { return backwardAccesses; }

    // Returns the interface of the view.
    inline const std::vector<const Iterator *>& getInterfaceIterators() const { return interface; }

    // Returns the map-reduce manipulations.
    inline const std::vector<const MapReduceOp *>& getManipulations() const { return manipulations; }

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
