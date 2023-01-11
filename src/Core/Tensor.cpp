#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "KAS/Core/BindingContext.hpp"
#include "KAS/Core/CodeGen.hpp"
#include "KAS/Core/Manipulation.hpp"
#include "KAS/Core/Tensor.hpp"
#include "KAS/Core/Iterator.hpp"
#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/Vector.hpp"
#include "KAS/Core/IteratorEvaluator.hpp"


namespace kas {

std::string Tensor::GetIndentSpaces(std::size_t indent) {
    return std::string(4 * indent, ' ');
}

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
    for (std::size_t i = 0; i < shape.size(); i++) {
        interface.emplace_back(std::make_shared<Iterator>(IteratorTransform { TensorStub { shared_from_this(), i } }, shape[i]));
    }
    return interface;
}

std::string Tensor::interfaceAccessToString(const BindingContext& ctx, const CodeGenContext& cgCtx) const {
    IteratorValuePrinter printer(ctx, cgCtx);
    return VectorToString(access, std::function([&](const std::shared_ptr<IteratorValue>& value) {
        return printer.toString(*value);
    }));
}

std::string Tensor::printNestedLoops(const BindingContext& ctx, const CodeGenContext& cgCtx) const {
    std::stringstream ss;
    auto [header, depth] = cgCtx.printOuterLoopsHeader(ctx);
    ss << header;
    ss << printInnerLoops(ctx, cgCtx, depth);
    ss << cgCtx.printOuterLoopsTail();
    return ss.str();
}

TensorStub::TensorStub(std::shared_ptr<Tensor> tensor, std::size_t index):
    tensor { std::move(tensor) },
    index { index }
{}

void TensorStub::setAccess(std::shared_ptr<IteratorValue> value) const {
    tensor->setAccess(std::move(value), index);
}

std::string PureTensor::printInnerLoops(const BindingContext& ctx, const CodeGenContext& cgCtx, std::size_t indent) const {
    // PureTensor does not need any loop.
    return GetIndentSpaces(indent) + actualAccessToString(ctx, cgCtx) + "\n";
}

void PureTensor::evaluateTensorAccess() {
    // no need to evaluate
}

std::string PureTensor::actualAccessToString(const BindingContext& ctx, const CodeGenContext& cgCtx) const {
    // They are actually the same.
    return std::string(cgCtx.getTensorName(tensorId)) + interfaceAccessToString(ctx, cgCtx);
}

Shape PureTensor::getShape() const {
    return shape;
}

std::string PureTensor::shapeToString(const BindingContext& ctx) const {
    return shape.toString(ctx);
}

std::string TensorView::printInnerLoops(const BindingContext& ctx, const CodeGenContext& cgCtx, std::size_t indent) const {
    std::stringstream ss;
    const auto recursion = [this, &ctx, &cgCtx, &ss](const auto& self, std::size_t manipId, std::size_t depth) -> std::string {
        if (manipId-- == 0) {
            std::stringstream innerSs;
            bool first = true;
            for (const auto& tensor: tensors) {
                if (first) {
                    first = false;
                } else {
                    innerSs << "*"; // TODO: other blending schemes other than multiplication
                }
                innerSs << tensor->actualAccessToString(ctx, cgCtx);
            }
            return innerSs.str();
        }

        auto& m = manipulations[manipId];
        auto name = cgCtx.getIteratorVariableName(m.iteratorVariableId);
        std::string tempName = "temp_" + std::string(name);

        // float temp_i_idx = 0;
        ss << GetIndentSpaces(depth);
        ss << "float " << tempName << " = 0;\n";

        // for (int i_idx = 0; i_idx < size_idx; i_idx++) {
        ss << GetIndentSpaces(depth);
        ss << "for (int " << name << " = 0; " << name << " < " << m.getIterator()->getSize()->toString(ctx) << "; " << name << "++) {\n";

        // Generate inner loops, and obtain the temporary variable.
        std::string inner = self(self, manipId, depth + 1);

        ss << GetIndentSpaces(depth + 1);
        // Can be other reduce than sum. TODO.
        ss << tempName << " += " << m.whatMap() << "(" << inner << ");\n";

        ss << GetIndentSpaces(depth);
        ss << "}\n";

        return tempName;
    };
    std::string lastTemp = recursion(recursion, manipulations.size(), indent);
    ss << GetIndentSpaces(indent);
    ss << "out" << interfaceAccessToString(ctx, cgCtx) << " = " << lastTemp << ";\n";
    return ss.str();
}

TensorView::TensorView(std::vector<std::shared_ptr<PureTensor>> tensors, std::shared_ptr<CodeGenContext> cgCtx):
    interface { [&tensors]() -> std::vector<std::shared_ptr<Iterator>> {
        std::vector<std::shared_ptr<Iterator>> res;
        for (const auto& tensor : tensors) {
            auto tensorInterface = tensor->getInterfaceStubs();
            res.insert(res.end(), tensorInterface.begin(), tensorInterface.end());
        }
        return res;
    }() },
    manipulations {},
    cgCtx { std::move(cgCtx) },
    tensors { std::move(tensors) },
    Tensor { std::vector<std::shared_ptr<IteratorValue>> {} }
{}

TensorView::TensorView(const Shape& shape, std::shared_ptr<CodeGenContext> cgCtx):
    TensorView { { std::make_shared<PureTensor>(cgCtx->addTensor("t"), shape) }, std::move(cgCtx) }
{}

void TensorView::finishConstruction() {
    access.resize(interface.size());
    for (auto& m: manipulations) {
        m.iteratorVariableId = cgCtx->addIteratorVariable(m.getIterator(), false);
    }
}

void TensorView::setDefaultInterfaceAccess() {
    auto defaultAccess = IteratorValue::DefaultAccessForShape(interface, *cgCtx);
    KAS_ASSERT(defaultAccess.size() == interface.size());
    access = std::move(defaultAccess);
}

std::size_t TensorView::size() const {
    return interface.size();
}
const std::shared_ptr<Iterator>& TensorView::operator[](std::size_t index) const {
    return interface.at(index);
}

void TensorView::replaceInterface(
    std::vector<std::size_t> drops,
    std::vector<std::pair<std::size_t, std::shared_ptr<Iterator>>> adds
) {
    auto replaced = ReplaceVector(interface, drops, adds);
    interface.swap(replaced);
}

void TensorView::addManipulation(Manipulation manipulation) {
    manipulations.emplace_back(std::move(manipulation));
}

const std::vector<std::shared_ptr<PureTensor>>& TensorView::getUnderlyingTensors() const {
    return tensors;
}

const std::vector<std::shared_ptr<Iterator>>& TensorView::getInterfaceIterators() const {
    return interface;
}

const std::vector<Manipulation>& TensorView::getManipulations() const {
    return manipulations;
}

void TensorView::evaluateTensorAccess() {
    KAS_ASSERT(access.size() == interface.size());
    KAS_ASSERT(std::all_of(access.begin(), access.end(), [](const auto& value) {
        return value != nullptr;
    }));
    IteratorEvaluator evaluator;
    evaluator.evaluateTensorAccess(*this);
    for (const auto& tensor: tensors) {
        tensor->evaluateTensorAccess();
    }
}

std::string TensorView::actualAccessToString(const BindingContext& ctx, const CodeGenContext& cgCtx) const {
    std::stringstream ss;
    // Here we assume the outer loops are exactly the interface iterators.
    ss << cgCtx.outerLoopIteratorsToString();
    for (const auto& m: manipulations) {
        ss << " with " << m.whatMap() << " mapped";
        ss << " with " << cgCtx.getIteratorVariableName(m.iteratorVariableId) << " " << m.whatReduce() << " reduced";
    }
    return ss.str();
}

std::vector<Shape> TensorView::getInputShapes() const {
    std::vector<Shape> res;
    for (const auto& tensor: tensors) {
        res.emplace_back(tensor->getShape());
    }
    return res;
}

Shape TensorView::getShape() const {
    std::vector<std::shared_ptr<Size>> sizes;
    for (const auto& iterator: interface) {
        sizes.emplace_back(iterator->getSize());
    }
    return Shape { sizes };
}

std::string TensorView::printNestedLoops(const BindingContext& ctx) const {
    return Tensor::printNestedLoops(ctx, *cgCtx);
}

std::string TensorView::shapeToString(const BindingContext &ctx) const {
    auto s1 = VectorToString(interface, std::function([&ctx](const std::shared_ptr<Iterator>& iterator) -> std::string {
        return iterator->getSize()->toString(ctx);
    }));
    if (!manipulations.empty()) {
        auto s2 = VectorToString(manipulations, std::function([&ctx](const Manipulation& m) -> std::string {
            return m.getIterator()->getSize()->toString(ctx);
        }));
        return s1 + " with reduced " + s2;
    }
    return s1;
}

void TensorView::addTransformDescription(std::string&& description) {
    transformDescriptions.emplace_back(std::move(description));
}

void TensorView::addIntermediateShape(std::string&& description) {
    intermediateShapes.emplace_back(std::move(description));
}

std::string TensorView::description(const BindingContext& ctx) const {
    KAS_ASSERT(transformDescriptions.size() + 1 == intermediateShapes.size());
    std::stringstream ss;
    for (std::size_t i = 0; i < transformDescriptions.size(); ++i) {
        ss << intermediateShapes[i] << '\n';
        ss << transformDescriptions[i] << '\n';
    }
    ss << intermediateShapes.back() << '\n';
    return ss.str();
}

} // namespace kas
