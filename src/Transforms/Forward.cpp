#include "KAS/Core/Tensor.hpp"
#include "KAS/Transforms/Forward.hpp"


namespace kas {

namespace Forward {

std::string Dimension::sizeToString() const {
    return getSize().toString(inner->getFactory().getBindingContext());
}

void Dimension::output(std::size_t index) {
    auto it = std::make_unique<Iterator>(index, getSize());
    set(it.get());
    getFactory().storeIterator(std::move(it));
}

void Dimension::reduce(std::size_t priority, MapReduce::MapType mapType, MapReduce::ReduceType reduceType) {
    auto op = std::make_unique<MapReduce>(priority, getSize(), mapType, reduceType);
    set(op.get());
    getFactory().storeMapReduce(std::move(op));
}

void Factory::storeIterator(std::unique_ptr<Iterator> iterator) {
    iterators.emplace_back(std::move(iterator));
}
void Factory::storeMapReduce(std::unique_ptr<MapReduce> mapReduce) {
    mapReduces.emplace_back(std::move(mapReduce));
}

std::vector<std::vector<BackwardDimension>> Factory::ForwardDimsToBackwardDims(const std::vector<std::vector<Dimension>>& tensors) {
    std::vector<std::vector<BackwardDimension>> backwardTensors(tensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        std::ranges::copy(tensors[i], std::back_inserter(backwardTensors[i]));
    }
    // Sort the dimensions by hash.
    // TODO: what if we change the order, due to performance considerations?
    for (std::size_t i = 1; i < tensors.size(); ++i) {
        std::ranges::sort(backwardTensors[i], BackwardDimension::HashLessThan{});
    }
    return backwardTensors;
}

TensorView& Factory::buildTensorView(const std::vector<std::vector<Dimension>>& tensors, TensorExpression blending) {
    KAS_ASSERT(!this->result, "Factory must not be used twice!");
    this->result = std::make_unique<TensorView>(ForwardDimsToBackwardDims(tensors), std::move(blending));
    return *this->result;
}

void MergeOp::onNotification(Factory& factory) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    auto op = factory.getStore().get<::kas::MergeOp>(outputDim, inputRhs.getSize());
    inputLhs.set(op->getInputL());
    inputRhs.set(op->getInputR());
}
Dimension MergeOp::Create(const Dimension& lhs, const Dimension& rhs) {
    auto op = std::unique_ptr<MergeOp> { new MergeOp { lhs, rhs } };
    auto output = Output::Create(lhs.getFactory(), lhs.getSize() * rhs.getSize(), std::move(op));
    return Dimension(std::move(output));
}

void ShareOp::onNotification(Factory& factory) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    auto op = factory.getStore().get<::kas::ShareOp>(outputDim);
    inputLhs.set(op->getInputL());
    inputRhs.set(op->getInputR());
}
Dimension ShareOp::Create(const Dimension& lhs, const Dimension& rhs) {
    auto op = std::unique_ptr<ShareOp> { new ShareOp { lhs, rhs } };
    KAS_ASSERT(lhs.getSize() == rhs.getSize(), "Shared dimensions must be the same size.");
    auto output = Output::Create(lhs.getFactory(), lhs.getSize(), std::move(op));
    return Dimension(std::move(output));
}

void ShiftOp::onNotification(Factory& factory) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    input.set(factory.getStore().get<::kas::ShiftOp>(outputDim, shift)->getInput());
}
Dimension ShiftOp::Create(const Dimension& input, int shift) {
    auto op = std::unique_ptr<ShiftOp> { new ShiftOp { input, shift } };
    auto output = Output::Create(input.getFactory(), input.getSize(), std::move(op));
    return Dimension(std::move(output));
}

void SplitOp::onNotification(Factory& factory) {
    if (outputLhs.lock()->evaluated() && outputRhs.lock()->evaluated()) {
        BackwardDimension outputLhsDim = outputLhs.lock()->get();
        BackwardDimension outputRhsDim = outputRhs.lock()->get();
        input.set(factory.getStore().get<::kas::SplitOp>(outputLhsDim, outputRhsDim)->getInput());
    }
}
std::pair<Dimension, Dimension> SplitOp::Create(const Dimension& input, const Size& block) {
    auto op = std::shared_ptr<SplitOp> { new SplitOp { input } };
    auto outputLhs = Output::Create(input.getFactory(), input.getSize() / block, op, Order::Left);
    auto outputRhs = Output::Create(input.getFactory(), block, op, Order::Right);
    return { Dimension(std::move(outputLhs)), Dimension(std::move(outputRhs)) };
}

void StrideOp::onNotification(Factory& factory) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    input.set(factory.getStore().get<::kas::StrideOp>(outputDim, stride)->getInput());
}
Dimension StrideOp::Create(const Dimension& input, const Size& stride) {
    auto op = std::unique_ptr<StrideOp> { new StrideOp { input, stride } };
    auto output = Output::Create(input.getFactory(), input.getSize() / stride, std::move(op));
    return Dimension(std::move(output));
}

void UnfoldOp::onNotification(Factory& factory) {
    if (outputLhs.lock()->evaluated() && outputRhs.lock()->evaluated()) {
        BackwardDimension outputLhsDim = outputLhs.lock()->get();
        BackwardDimension outputRhsDim = outputRhs.lock()->get();
        input.set(factory.getStore().get<::kas::UnfoldOp>(outputLhsDim, outputRhsDim)->getInput());
    }
}
std::pair<Dimension, Dimension> UnfoldOp::Create(const Dimension& input, const Size& window) {
    auto op = std::shared_ptr<UnfoldOp> { new UnfoldOp { input } };
    auto outputLhs = Output::Create(input.getFactory(), input.getSize(), op, Order::Left);
    auto outputRhs = Output::Create(input.getFactory(), window, op, Order::Right);
    return { Dimension(std::move(outputLhs)), Dimension(std::move(outputRhs)) };
}

} // namespace Forward

} // namespace kas
