#include "KAS/Transforms/Forward.hpp"


namespace kas {

namespace Forward {

std::unique_ptr<Iterator> Dimension::output(std::size_t index) {
    auto it = std::make_unique<Iterator>(index, getSize());
    set(it.get());
    return it;
}

std::unique_ptr<MapReduceOp> Dimension::reduce(std::size_t priority, MapReduceOp::MapType mapType, MapReduceOp::ReduceType reduceType) {
    auto op = std::make_unique<MapReduceOp>(priority, getSize(), mapType, reduceType);
    set(op.get());
    return op;
}

void MergeOp::onNotification(DimensionStore& store) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    auto op = store.get<::kas::MergeOp>(outputDim, inputRhs.getSize());
    auto [inLhs, inRhs] = op->getInputs();
    inputLhs.set(inLhs);
    inputRhs.set(inRhs);
}
Dimension MergeOp::Create(const Dimension& lhs, const Dimension& rhs) {
    auto op = std::unique_ptr<MergeOp> { new MergeOp { lhs, rhs } };
    auto output = Output::Create(lhs.getStore(), lhs.getSize() * rhs.getSize(), std::move(op));
    return Dimension(std::move(output));
}

void ShareOp::onNotification(DimensionStore& store) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    auto op = store.get<::kas::ShareOp>(outputDim);
    auto [inLhs, inRhs] = op->getInputs();
    inputLhs.set(inLhs);
    inputRhs.set(inRhs);
}
Dimension ShareOp::Create(const Dimension& lhs, const Dimension& rhs) {
    auto op = std::unique_ptr<ShareOp> { new ShareOp { lhs, rhs } };
    KAS_ASSERT(lhs.getSize() == rhs.getSize(), "Shared dimensions must be the same size.");
    auto output = Output::Create(lhs.getStore(), lhs.getSize(), std::move(op));
    return Dimension(std::move(output));
}

void ShiftOp::onNotification(DimensionStore& store) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    input.set(store.get<::kas::ShiftOp>(outputDim, shift)->getInput());
}
Dimension ShiftOp::Create(const Dimension& input, int shift) {
    auto op = std::unique_ptr<ShiftOp> { new ShiftOp { input, shift } };
    auto output = Output::Create(input.getStore(), input.getSize(), std::move(op));
    return Dimension(std::move(output));
}

void SplitOp::onNotification(DimensionStore& store) {
    if (outputLhs.lock()->evaluated() && outputRhs.lock()->evaluated()) {
        BackwardDimension outputLhsDim = outputLhs.lock()->get();
        BackwardDimension outputRhsDim = outputRhs.lock()->get();
        input.set(store.get<::kas::SplitOp>(outputLhsDim, outputRhsDim)->getInput());
    }
}
std::pair<Dimension, Dimension> SplitOp::Create(const Dimension& input, const Size& block) {
    auto op = std::shared_ptr<SplitOp> { new SplitOp { input } };
    auto outputLhs = Output::Create(input.getStore(), input.getSize() / block, op, Order::Left);
    auto outputRhs = Output::Create(input.getStore(), block, op, Order::Right);
    return { Dimension(std::move(outputLhs)), Dimension(std::move(outputRhs)) };
}

void StrideOp::onNotification(DimensionStore& store) {
    KAS_ASSERT(output.lock()->evaluated());
    BackwardDimension outputDim = output.lock()->get();
    input.set(store.get<::kas::StrideOp>(outputDim, stride)->getInput());
}
Dimension StrideOp::Create(const Dimension& input, const Size& stride) {
    auto op = std::unique_ptr<StrideOp> { new StrideOp { input, stride } };
    auto output = Output::Create(input.getStore(), input.getSize() / stride, std::move(op));
    return Dimension(std::move(output));
}

void UnfoldOp::onNotification(DimensionStore& store) {
    if (outputLhs.lock()->evaluated() && outputRhs.lock()->evaluated()) {
        BackwardDimension outputLhsDim = outputLhs.lock()->get();
        BackwardDimension outputRhsDim = outputRhs.lock()->get();
        input.set(store.get<::kas::UnfoldOp>(outputLhsDim, outputRhsDim)->getInput());
    }
}
std::pair<Dimension, Dimension> UnfoldOp::Create(const Dimension& input, const Size& window) {
    auto op = std::shared_ptr<UnfoldOp> { new UnfoldOp { input } };
    auto outputLhs = Output::Create(input.getStore(), input.getSize(), op, Order::Left);
    auto outputRhs = Output::Create(input.getStore(), window, op, Order::Right);
    return { Dimension(std::move(outputLhs)), Dimension(std::move(outputRhs)) };
}

} // namespace Forward

} // namespace kas
