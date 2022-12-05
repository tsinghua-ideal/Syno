#include <functional>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"
#include "KAS/Core/IteratorEvaluator.hpp"


namespace kas {

void PureTensor::setAccess(std::shared_ptr<IteratorValue> value, int index) {
    KAS_ASSERT(index < access.size());
    access[index] = std::move(value);
}

std::vector<std::shared_ptr<Iterator>> PureTensor::getInterface() {
    std::vector<std::shared_ptr<Iterator>> interface;
    interface.reserve(shape.size());
    for (int i = 0; i < shape.size(); i++) {
        interface.push_back(std::make_shared<Iterator>(IteratorTransform { TensorStub { shared_from_this(), i } }, shape[i]));
    }
    return interface;
}

TensorView PureTensor::buildTensorView() {
    return TensorView { shared_from_this() };
}

std::string PureTensor::accessToString() const {
    return VectorToString(access, std::function([](const std::shared_ptr<IteratorValue>& value) {
        return value->content;
    }));
}

std::string PureTensor::shapeToString(const BindingContext& ctx) const {
    return shape.toString(ctx);
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
    interface { tensor->getInterface() },
    manipulations {},
    tensor { std::move(tensor) }
{}

TensorView::TensorView(const Shape& shape):
    TensorView { std::make_shared<PureTensor>(shape) }
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

void TensorView::addManipulation(Manipulation manipulation) {
    manipulations.push_back(std::move(manipulation));
}

std::shared_ptr<PureTensor> TensorView::getUnderlyingTensor() const {
    return tensor;
}

const std::vector<std::shared_ptr<Iterator>>& TensorView::getInterfaceIterators() const {
    return interface;
}

std::vector<std::shared_ptr<Iterator>> TensorView::getReducedIterators() const {
    std::vector<std::shared_ptr<Iterator>> reducedIterators {};
    for (const auto& manipulation: manipulations) {
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ReduceManipulation>) {
                reducedIterators.push_back(arg.iterator);
            }
        }, manipulation);
    }
    return reducedIterators;
}

std::vector<MapManipulation> TensorView::getMaps() const {
    std::vector<MapManipulation> maps {};
    for (const auto& manipulation: manipulations) {
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, MapManipulation>) {
                maps.push_back(arg);
            }
        }, manipulation);
    }
    return maps;
}

std::vector<std::shared_ptr<Iterator>> TensorView::getAllIterators() const {
    std::vector<std::shared_ptr<Iterator>> iterators(interface);
    iterators.reserve(interface.size() + manipulations.size());
    auto reducedIterators = getReducedIterators();
    iterators.insert(iterators.end(), reducedIterators.begin(), reducedIterators.end());
    return iterators;
}

void TensorView::evaluateTensorAccess(const BindingContext& ctx) const {
    IteratorEvaluator evaluator { ctx };
    evaluator.evaluateTensorAccess(*this);
}

std::string TensorView::accessToString() const {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < interface.size(); i++) {
        if (i != 0) {
            ss << ",";
        }
        ss << "i_" << i;
    }
    ss << "]";
    auto reducedIterators = getReducedIterators();
    if (!reducedIterators.empty()) {
        ss << " with reduced [";
        for (int i = interface.size(); i < interface.size() + reducedIterators.size(); i++) {
            if (i != interface.size()) {
                ss << ",";
            }
            ss << "i_" << i;
        }
        ss << "]";
    }
    auto maps = getMaps();
    if (!maps.empty()) {
        ss << " with mapped ";
        ss << VectorToString(maps, std::function([](const MapManipulation& map) {
            return map.what();
        }));
    }
    return ss.str();
}

std::string TensorView::shapeToString(const BindingContext &ctx) const {
    auto mapper = std::function([&ctx](const std::shared_ptr<Iterator>& iterator) -> std::string {
        return iterator->getSize()->toString(ctx);
    });
    auto s1 = VectorToString(interface, mapper);
    auto reducedIterators = getReducedIterators();
    if (!reducedIterators.empty()) {
        auto s2 = VectorToString(reducedIterators, mapper);
        return s1 + " with reduced " + s2;
    }
    return s1;
}

} // namespace kas
