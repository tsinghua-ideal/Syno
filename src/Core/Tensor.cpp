#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"
#include "KAS/Core/IteratorEvaluator.hpp"


namespace kas {

void Tensor::setAccess(std::shared_ptr<IteratorValue> value, std::size_t index) {
    KAS_ASSERT(index < access.size());
    access[index] = std::move(value);
}

std::shared_ptr<IteratorValue> Tensor::getAccess(std::size_t index) const {
    KAS_ASSERT(index < access.size());
    return access[index];
}

std::vector<std::shared_ptr<Iterator>> Tensor::getInterfaceStubs() {
    std::vector<std::shared_ptr<Iterator>> interface;
    const auto shape = getShape();
    interface.reserve(shape.size());
    for (int i = 0; i < shape.size(); i++) {
        interface.push_back(std::make_shared<Iterator>(IteratorTransform { TensorStub { shared_from_this(), i } }, shape[i]));
    }
    return interface;
}

std::string Tensor::interfaceAccessToString(const BindingContext& ctx) const {
    IteratorValuePrinter printer(ctx);
    return VectorToString(access, std::function([&](const std::shared_ptr<IteratorValue>& value) {
        return printer.toString(*value);
    }));
}

TensorStub::TensorStub(std::shared_ptr<Tensor> tensor, int index):
    tensor { std::move(tensor) },
    index { index }
{}

void TensorStub::setAccess(std::shared_ptr<IteratorValue> value) const {
    tensor->setAccess(std::move(value), index);
}

void PureTensor::evaluateTensorAccess(BindingContext& ctx) {
    // no need to evaluate
}

std::string PureTensor::actualAccessToString(const BindingContext& ctx) const {
    // They are actually the same.
    return std::string(ctx.getTensorName(tensorId)) + interfaceAccessToString(ctx);
}

Shape PureTensor::getShape() const {
    return shape;
}

std::string PureTensor::shapeToString(const BindingContext& ctx) const {
    return shape.toString(ctx);
}

TensorView::TensorView(std::shared_ptr<Tensor> tensor):
    interface { tensor->getInterfaceStubs() },
    manipulations {},
    tensor { std::move(tensor) },
    Tensor { std::vector<std::shared_ptr<IteratorValue>> {} }
{}

TensorView::TensorView(const Shape& shape, BindingContext& ctx):
    TensorView { std::make_shared<PureTensor>(ctx.addTensor("t"), shape) }
{}

void TensorView::finishConstruction() {
    access.resize(interface.size());
    reducedAccess.resize(getReducedIterators().size());
}

void TensorView::setAccesses(std::vector<std::shared_ptr<IteratorValue>> accesses) {
    KAS_ASSERT(accesses.size() == interface.size());
    access = std::move(accesses);
}

void TensorView::setDefaultAccesses(BindingContext& ctx) {
    setAccesses(IteratorValue::DefaultAccessForShape(getShape(), ctx));
}

void TensorView::setReducedAccess(std::shared_ptr<IteratorValue> value, std::size_t index) {
    reducedAccess[index] = std::move(value);
}

std::size_t TensorView::size() const {
    return interface.size();
}
const std::shared_ptr<Iterator>& TensorView::operator[](std::size_t index) const {
    return interface.at(index);
}

void TensorView::replaceInterface(
    std::vector<int> drops,
    std::vector<std::pair<int, std::shared_ptr<Iterator>>> adds
) {
    auto replaced = ReplaceVector(interface, drops, adds);
    interface.swap(replaced);
}

void TensorView::addManipulation(Manipulation manipulation) {
    manipulations.push_back(std::move(manipulation));
}

std::shared_ptr<Tensor> TensorView::getUnderlyingTensor() const {
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

void TensorView::evaluateTensorAccess(BindingContext& ctx) {
    IteratorEvaluator evaluator { ctx };
    evaluator.evaluateTensorAccess(*this);
    tensor->evaluateTensorAccess(ctx);
}

std::string TensorView::actualAccessToString(const BindingContext& ctx) const {
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
        for (int i = 0; i < reducedIterators.size(); i++) {
            if (i != 0) {
                ss << ",";
            }
            ss << "ri_" << i;
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

Shape TensorView::getShape() const {
    std::vector<std::shared_ptr<Size>> sizes;
    for (const auto& iterator: interface) {
        sizes.push_back(iterator->getSize());
    }
    return Shape { sizes };
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
