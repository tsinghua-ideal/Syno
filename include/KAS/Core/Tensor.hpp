#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Manipulation.hpp"


namespace kas {

class Iterator;
class IteratorValue;

class Tensor: public std::enable_shared_from_this<Tensor> {
protected:
    std::vector<std::shared_ptr<IteratorValue>> access;

public:
    template<typename T>
    Tensor(T&& access):
        access(std::forward<T>(access))
    {}
    void setAccess(std::shared_ptr<IteratorValue> value, std::size_t index);
    std::shared_ptr<IteratorValue> getAccess(std::size_t index) const;
    virtual void evaluateTensorAccess(BindingContext& ctx) = 0;
    std::vector<std::shared_ptr<Iterator>> getInterfaceStubs();
    std::string interfaceAccessToString(const BindingContext& ctx) const;
    // Requires evaluateTensorAccess to be called first
    virtual std::string actualAccessToString(const BindingContext& ctx) const = 0;
    virtual Shape getShape() const = 0;
    virtual std::string shapeToString(const BindingContext& ctx) const = 0;
    template<typename Derived>
    std::shared_ptr<Derived> shared_from_base() {
        return std::dynamic_pointer_cast<Derived>(shared_from_this());
    }
};

class TensorStub {
public:
    const std::shared_ptr<Tensor> tensor;
    const int index;
    TensorStub(std::shared_ptr<Tensor> tensor, int index);

    void setAccess(std::shared_ptr<IteratorValue> value) const;
};

// PureTensor must be created with std::make_shared()!
class PureTensor: public Tensor {
protected:
    std::size_t tensorId;
    Shape shape;

public:
    // Initialize access as nullptr
    template<typename T>
    PureTensor(std::size_t tensorId, T&& shape):
        Tensor { std::vector<std::shared_ptr<IteratorValue>>(shape.size(), nullptr) },
        tensorId { tensorId },
        shape { std::forward<T>(shape) }
    {}
    void evaluateTensorAccess(BindingContext& ctx) override;
    std::string actualAccessToString(const BindingContext& ctx) const override;
    Shape getShape() const override;
    std::string shapeToString(const BindingContext& ctx) const override;
};

class TensorView: public Tensor {
protected:
    std::vector<std::shared_ptr<Iterator>> interface;
    std::vector<Manipulation> manipulations;
    std::shared_ptr<Tensor> tensor;

    std::vector<std::shared_ptr<IteratorValue>> reducedAccess;

    friend class FinalizeShapeOp;

public:
    TensorView(std::shared_ptr<Tensor> tensor);
    TensorView(const Shape& shape, BindingContext& ctx);
    // This sets the size of access
    void finishConstruction();
    // Set accesses once and for all
    void setAccesses(std::vector<std::shared_ptr<IteratorValue>> accesses);
    // Use BindingContext to generate accesses "i_0, i_1, ..."
    void setDefaultAccesses(BindingContext& ctx);
    // Set access for reduced iterators
    void setReducedAccess(std::shared_ptr<IteratorValue> value, std::size_t index);

    std::size_t size() const;
    const std::shared_ptr<Iterator>& operator[](std::size_t index) const;

    // drops and adds must be sorted by index
    void replaceInterface(
        std::vector<int> drops,
        std::vector<std::pair<int, std::shared_ptr<Iterator>>> adds
    );

    // A manipulation is a transform of the data in a tensor, not just the way of accessing it. This includes Map and Reduce.
    void addManipulation(Manipulation manipulation);

    // Returns the underlying tensor underneath the view.
    std::shared_ptr<Tensor> getUnderlyingTensor() const;

    // Returns the interface of the view.
    const std::vector<std::shared_ptr<Iterator>>& getInterfaceIterators() const;

    // Returns reduced iterators.
    std::vector<std::shared_ptr<Iterator>> getReducedIterators() const;

    // Returns maps.
    std::vector<MapManipulation> getMaps() const;

    // This returns all iterators, including interface and reduced iterators.
    std::vector<std::shared_ptr<Iterator>> getAllIterators() const;

    // Evaluates all accesses to the underlying tensor.
    void evaluateTensorAccess(BindingContext& ctx) override;

    // Returns something like "[i_0,i_1] with reduced [i_2]". When Blending is implemented, an additional argument may be passed. TODO
    std::string actualAccessToString(const BindingContext &ctx) const override;

    Shape getShape() const override;

    // Returns the shapes of all iterators, including reduced iterators.
    std::string shapeToString(const BindingContext &ctx) const override;
};

} // namespace kas
