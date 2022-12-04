#pragma once

#include <memory>
#include <vector>

#include "KAS/Core/Shape.hpp"


namespace kas {

class Iterator;
class IteratorValue;

// Only handles a single tensor. Multiple tensors (including TensorView's) can be blended into a single tensor. TODO
class PureTensor: public std::enable_shared_from_this<PureTensor> {
public:
    std::vector<std::shared_ptr<IteratorValue>> access;
    Shape shape;
    // Initialize access as nullptr
    PureTensor(const Shape& shape);

    void setAccess(std::shared_ptr<IteratorValue> value, int index);

    std::vector<std::shared_ptr<Iterator>> getInterface();
};

class TensorStub {
public:
    const std::shared_ptr<PureTensor> tensor;
    const int index;
    TensorStub(std::shared_ptr<PureTensor> tensor, int index);

    void setAccess(std::shared_ptr<IteratorValue> value) const;
};

// Reduce, Map
class Manipulation {

};

class TensorView {
public:
    std::vector<std::shared_ptr<Iterator>> interface;
    std::vector<Manipulation> manipulations;
    const std::shared_ptr<PureTensor> tensor;

    TensorView(std::shared_ptr<PureTensor> tensor);

    size_t size() const;
    const std::shared_ptr<Iterator>& operator[](int index) const;

    // drops and adds must be sorted by index
    void replaceInterface(
        const std::vector<int>& drops,
        const std::vector<std::pair<int, std::shared_ptr<Iterator>>>& adds
    );

    std::string shapeToString(const BindingContext& ctx) const;
};

} // namespace kas
