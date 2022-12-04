#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"
#include <sstream>


namespace kas {

void PureTensor::setAccess(std::shared_ptr<IteratorValue> value, int index) {
    KAS_ASSERT(index < access.size());
    access[index] = std::move(value);
}

std::vector<std::shared_ptr<Iterator>> PureTensor::getInterface() {
    std::vector<std::shared_ptr<Iterator>> interface;
    interface.reserve(shape.size());
    for (int i = 0; i < shape.size(); i++) {
        interface.push_back(std::make_shared<Iterator>(IteratorTransform { TensorStub { shared_from_this(), i } }));
    }
    return interface;
}

TensorStub::TensorStub(std::shared_ptr<PureTensor> tensor, int index):
    tensor { std::move(tensor) },
    index { index }
{}

void TensorStub::setAccess(std::shared_ptr<IteratorValue> value) const {
    tensor->setAccess(std::move(value), index);
}

PureTensor::PureTensor(const Shape& shape):
    access { std::vector<std::shared_ptr<IteratorValue>>(shape.size(), nullptr) },
    shape { shape }
{}

TensorView::TensorView(std::shared_ptr<PureTensor> tensor):
    interface { std::move(tensor->getInterface()) },
    manipulations {},
    tensor { std::move(tensor) }
{}

size_t TensorView::size() const {
    return interface.size();
}
const std::shared_ptr<Iterator>& TensorView::operator[](int index) const {
    return interface[index];
}

void TensorView::replaceInterface(
    const std::vector<int>& drops,
    const std::vector<std::pair<int, std::shared_ptr<Iterator>>>& adds
) {
    auto replaced = ReplaceVector(interface, drops, adds);
    interface.swap(replaced);
}

std::string TensorView::shapeToString(const BindingContext &ctx) const {
    std::stringstream ss;
    // TODO
    return "TODO";
}

} // namespace kas
