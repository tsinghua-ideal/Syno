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
    friend class GraphvizGen;
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
};

struct AbstractAccess {
    // -1 for output tensor, otherwise index of input tensors.
    int position;
    std::size_t outerLoopsCount;
    Shape innerLoopsShape;
    std::vector<std::vector<IteratorValue>> inputs;
    std::vector<IteratorValue> output;
};

class TensorView {
    friend class HalideGen;
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

    // Returns the outer loop initializers and depth of the loops.
    std::string printInnerLoops(const BindingContext& ctx, std::size_t indent, std::string_view outputName) const;

public:
    // Build the tensor from iterator DAG.
    TensorView(const std::vector<std::vector<Dimension>>& tensors);

    inline IteratorShapeView getShape() const { return IteratorShapeView(interface); }
    inline ReduceIteratorShapeView getReduceShape() const { return ReduceIteratorShapeView(manipulations); }

    // Returns the underlying tensor underneath the view.
    inline const std::vector<PureTensor>& getUnderlyingTensors() const { return tensors; }

    // This blends all the tensors together.
    std::string fusedInputToString(const BindingContext& ctx) const;

    // Returns the interface of the view.
    inline const std::vector<const Iterator *>& getInterfaceIterators() const { return interface; }

    // Returns the map-reduce manipulations.
    inline const std::vector<const MapReduceOp *>& getManipulations() const { return manipulations; }

    // The standard interface, i.e., the outer loops.
    std::string interfaceAccessToString(const BindingContext& ctx) const;

    // The inner reduction loops.
    std::string reduceAccessToString(const BindingContext& ctx) const;

    // Returns something like "[i_0,i_1,ri_0] with ri_0 Sum reduced". This is just the combined above two.
    std::string actualAccessToString(const BindingContext &ctx) const;

    std::string printNestedLoops(const BindingContext& ctx, std::string_view outputName = std::string_view("out")) const;
};

} // namespace kas
