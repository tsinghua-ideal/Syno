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
    std::vector<IteratorValue> access;
public:
    PureTensor(auto&& name, auto&& dims, auto&& access):
        name { std::forward<decltype(name)>(name) },
        dims { std::forward<decltype(dims)>(dims) },
        access { std::forward<decltype(access)>(access) }
    {}
    inline const std::string& getName() const { return name; }
    inline ShapeView getShape() const { return ShapeView(dims); }
    inline const std::vector<IteratorValue>& getAccess() const { return access; }
    std::string shapeToString(const BindingContext& ctx) const;
    std::string accessToString(const BindingContext& ctx) const;
};

class TensorView {
protected:
    // The interface iterators.
    std::vector<const Iterator *> interface;
    // Define the corresponding shape view for interface.
    using IteratorShapeView = AbstractShape<const std::vector<const Iterator *>&, [](const Iterator * const *ptr) -> const Size& { return (*ptr)->size(); }>;
    // The map-reduce iterators.
    std::vector<const MapReduceOp *> manipulations;
    using ReduceIteratorShapeView = AbstractShape<const std::vector<const MapReduceOp *>&, [](const MapReduceOp * const *ptr) -> const Size& { return (*ptr)->size(); }>;
    // How to blend the tensors? TODO
    std::vector<PureTensor> tensors;

    // Returns the outer loop initializers and depth of the loops.
    std::string printInnerLoops(const BindingContext& ctx, std::size_t indent, std::string_view outputName) const;

public:
    // Build the tensor from iterator DAG.
    TensorView(const std::vector<std::vector<Dimension>>& tensors);

    inline std::size_t size() const { return interface.size(); }
    inline const Iterator *operator[](std::size_t index) const { return interface[index]; }

    inline IteratorShapeView getShape() const { return IteratorShapeView(interface); }

    // Returns the underlying tensor underneath the view.
    inline const std::vector<PureTensor>& getUnderlyingTensors() const { return tensors; }

    // Returns the interface of the view.
    inline const std::vector<const Iterator *>& getInterfaceIterators() const { return interface; }

    // Returns the map-reduce manipulations.
    inline const std::vector<const MapReduceOp *>& getManipulations() const { return manipulations; }

    std::string shapeToString(const BindingContext& ctx) const;

    // The standard interface.
    std::string interfaceAccessToString(const BindingContext& ctx) const;

    // Returns something like "[i_0,i_1] with reduced [i_2]".
    std::string actualAccessToString(const BindingContext &ctx) const;

    std::string printNestedLoops(const BindingContext& ctx, std::string_view outputName = std::string_view("out")) const;
};

} // namespace kas
