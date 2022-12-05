#pragma once

#include <memory>
#include <variant>
#include <vector>

#include "KAS/Core/Shape.hpp"
#include "KAS/Core/Manipulation.hpp"


namespace kas {

class Iterator;
class IteratorValue;

class TensorView;

// Only handles a single tensor. Multiple tensors (including TensorView's) can be blended into a single tensor. TODO
// PureTensor must be created with std::make_shared()!
class PureTensor: public std::enable_shared_from_this<PureTensor> {
public:
    std::vector<std::shared_ptr<IteratorValue>> access;
    Shape shape;
    // Initialize access as nullptr
    PureTensor(const Shape& shape);

    void setAccess(std::shared_ptr<IteratorValue> value, int index);

    std::vector<std::shared_ptr<Iterator>> getInterface();

    TensorView buildTensorView();

    std::string accessToString() const;

    std::string shapeToString(const BindingContext& ctx) const;
};

class TensorStub {
public:
    const std::shared_ptr<PureTensor> tensor;
    const int index;
    TensorStub(std::shared_ptr<PureTensor> tensor, int index);

    void setAccess(std::shared_ptr<IteratorValue> value) const;
};

class TensorView {
protected:
    std::vector<std::shared_ptr<Iterator>> interface;
    std::vector<Manipulation> manipulations;
    const std::shared_ptr<PureTensor> tensor;

public:
    TensorView(std::shared_ptr<PureTensor> tensor);
    TensorView(const Shape& shape);

    size_t size() const;
    const std::shared_ptr<Iterator>& operator[](int index) const;

    // drops and adds must be sorted by index
    void replaceInterface(
        std::vector<int> drops,
        std::vector<std::pair<int, std::shared_ptr<Iterator>>> adds
    );

    // A manipulation is a transform of the data in a tensor, not just the way of accessing it. This includes Map and Reduce.
    void addManipulation(Manipulation manipulation);

    // Returns the underlying tensor underneath the view.
    std::shared_ptr<PureTensor> getUnderlyingTensor() const;

    // Returns the interface of the view.
    const std::vector<std::shared_ptr<Iterator>>& getInterfaceIterators() const;

    // Returns reduced iterators.
    std::vector<std::shared_ptr<Iterator>> getReducedIterators() const;

    // Returns maps.
    std::vector<MapManipulation> getMaps() const;

    // This returns all iterators, including interface and reduced iterators.
    std::vector<std::shared_ptr<Iterator>> getAllIterators() const;

    // Evaluates all accesses to the underlying tensor.
    void evaluateTensorAccess(const BindingContext& ctx) const;

    // Returns something like "[i_0,i_1] with reduced [i_2]". When Blending is implemented, an additional argument may be passed. TODO
    std::string accessToString() const;

    // Returns the shapes of all iterators, including reduced iterators.
    std::string shapeToString(const BindingContext &ctx) const;
};

} // namespace kas
