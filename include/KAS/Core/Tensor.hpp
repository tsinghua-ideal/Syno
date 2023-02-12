#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Manipulation.hpp"


namespace kas {

class Iterator;
struct IteratorValue;
class Tensor: public std::enable_shared_from_this<Tensor> {
    friend class HalideGen;
    friend class TensorView;

protected:
    std::vector<IteratorValue> access;

    static std::string GetIndentSpaces(std::size_t indent);
    virtual std::string printInnerLoops(const BindingContext& ctx, const CodeGenContext& cgCtx, std::size_t indent) const = 0;

public:
    Tensor(auto&& access):
        access(std::forward<decltype(access)>(access))
    {}
    void setAccess(IteratorValue value, std::size_t index);
    IteratorValue getAccess(std::size_t index) const;
    virtual void evaluateTensorAccess() = 0;
    std::vector<std::shared_ptr<Iterator>> getInterfaceStubs();
    // The standard interface.
    std::string interfaceAccessToString(const BindingContext& ctx, const CodeGenContext& cgCtx) const;
    // This is just some arbitrary description. Requires evaluateTensorAccess to be called first.
    virtual std::string actualAccessToString(const BindingContext& ctx, const CodeGenContext& cgCtx) const = 0;
    virtual Shape getShape() const = 0;
    virtual std::string shapeToString(const BindingContext& ctx) const = 0;
    std::string printNestedLoops(const BindingContext& ctx, const CodeGenContext& cgCtx) const;
    template<typename Derived>
    std::shared_ptr<Derived> shared_from_base() {
        return std::dynamic_pointer_cast<Derived>(shared_from_this());
    }
};

class TensorStub {
public:
    const std::shared_ptr<Tensor> tensor;
    const std::size_t index;
    TensorStub(std::shared_ptr<Tensor> tensor, std::size_t index);

    void setAccess(IteratorValue value) const;
};

// PureTensor must be created with std::make_shared()!
class PureTensor: public Tensor {
    friend class HalideGen;

protected:
    std::size_t tensorId;
    const Shape shape;

    std::string printInnerLoops(const BindingContext& ctx, const CodeGenContext& cgCtx, std::size_t indent) const override;

public:
    // Initialize access as nullptr
    PureTensor(std::size_t tensorId, auto&& shape):
        Tensor { std::vector<IteratorValue>(shape.size(), IteratorValue()) },
        tensorId { tensorId },
        shape { std::forward<decltype(shape)>(shape) }
    {}
    void evaluateTensorAccess() override;
    std::string actualAccessToString(const BindingContext& ctx, const CodeGenContext& cgCtx) const override;
    Shape getShape() const override;
    const Shape& getShapeRef() const;
    std::string shapeToString(const BindingContext& ctx) const override;
};

class TensorView: public Tensor {
    friend class FinalizeShapeOp;
    friend class HalideGen;

protected:
    // The interface iterators.
    std::vector<std::shared_ptr<Iterator>> interface;
    // The map-reduce manipulations.
    std::vector<Manipulation> manipulations;
    // How to blend the tensors? TODO
    std::vector<std::shared_ptr<PureTensor>> tensors;

    std::shared_ptr<CodeGenContext> cgCtx;

    std::string printInnerLoops(const BindingContext& ctx, const CodeGenContext& cgCtx, std::size_t indent) const override;

public:
    // Explicitly control the underlying tensors,
    TensorView(std::vector<std::shared_ptr<PureTensor>> tensors, std::shared_ptr<CodeGenContext> cgCtx);
    // Or for convenience just create a new tensor all by default.
    TensorView(const Shape& shape, std::shared_ptr<CodeGenContext> cgCtx);
    // This sets the size of interface access for filling, and assigns iterator variables to reduced dimensions.
    void finishConstruction();
    // Use CodeGenContext to generate access "i_0, i_1, ...".
    void setDefaultInterfaceAccess();

    std::size_t size() const;
    const std::shared_ptr<Iterator>& operator[](std::size_t index) const;

    void replaceInterface(
        std::vector<std::size_t> drops,
        std::vector<std::pair<std::size_t, std::shared_ptr<Iterator>>> adds
    );

    // A manipulation is a transform of the data in a tensor, not just the way of accessing it. This is just a Map bundled with a Reduce.
    void addManipulation(Manipulation manipulation);

    // Returns the underlying tensor underneath the view.
    const std::vector<std::shared_ptr<PureTensor>>& getUnderlyingTensors() const;

    // Returns the interface of the view.
    const std::vector<std::shared_ptr<Iterator>>& getInterfaceIterators() const;

    // Returns the map-reduce manipulations.
    const std::vector<Manipulation>& getManipulations() const;

    // Evaluates all accesses to the underlying tensor.
    void evaluateTensorAccess() override;

    // Returns something like "[i_0,i_1] with reduced [i_2]".
    std::string actualAccessToString(const BindingContext &ctx, const CodeGenContext& cgCtx) const override;

    Shape getFusedInputShapes() const;

    Shape getShape() const override;

    std::string printNestedLoops(const BindingContext& ctx) const;

    // Returns the shapes of all iterators, including reduced iterators.
    std::string shapeToString(const BindingContext &ctx) const override;
};

} // namespace kas
