#include "KAS/Core/Autodiff.hpp"


namespace kas {

namespace Derivative {

void DimensionGraphBuilder::visit(const Iterator& dim) {
    extra.outputIterators.push_back(&dim);
}

void DimensionGraphBuilder::visit(const MapReduceOp& dim) {
    extra.mapReduceIterators.push_back(&dim);
}

void DimensionGraphBuilder::visit(const RepeatLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { op };
    visit(op->output);
}

void DimensionGraphBuilder::visit(const SplitLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { std::make_pair(op, Order::Left) };
    visit(op->outputLhs);
    parent = { std::make_pair(op, Order::Right) };
    visit(op->outputRhs);
}

void DimensionGraphBuilder::visit(const MergeLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { op };
    visit(op->output);
}

void DimensionGraphBuilder::visit(const Dimension& dim) {
    auto [it, inserted] = extra.theOtherEndOfEdge.insert({dim, parent});
    if (inserted) {
        DimVisitor::visit(dim);
    }
}

void DimensionGraphBuilder::add(const Dimension& dim) {
    parent = { std::monostate{} };
    visit(dim);
}
AdditionalMetadata DimensionGraphBuilder::build() {
    std::ranges::sort(extra.outputIterators, [](const Iterator *lhs, const Iterator *rhs) {
        return lhs->getIndex() < rhs->getIndex();
    });
    std::ranges::sort(extra.mapReduceIterators, [](const MapReduceOp *lhs, const MapReduceOp *rhs) {
        return lhs->getPriority() < rhs->getPriority();
    });
    return std::move(extra);
}

void DimensionEvaluator::assign(const Dimension& dim, IteratorValue value) {
    auto [it, _] = values.insert_or_assign(dim, IteratorValue{});
    KAS_ASSERT(!it->second);
    it->second = value;
    remaining.erase(dim);
    walkDown<true>(dim);
    walkUp<true>(dim);
}

DimensionEvaluator::DimensionEvaluator(const std::map<Dimension, OpAbove, Dimension::AddressLessThan>& theOtherEndOfEdge):
    theOtherEndOfEdge { theOtherEndOfEdge }
{
    // Initialize the remaining set.
    std::ranges::copy(theOtherEndOfEdge | std::views::transform([](auto&& kv) { return kv.first; }), std::inserter(remaining, remaining.end()));
}

void DimensionEvaluator::makeVar(const Dimension& dim) {
    outerLoops.emplace_back(VariableValueNode::Create(false, outerLoops.size(), "i_" + std::to_string(outerLoops.size())));
    assign(dim, outerLoops.back());
}

void DimensionEvaluator::makeVars(const std::vector<Dimension>& dims) {
    for (const auto& dim: dims) {
        makeVar(dim);
    }
}

void DimensionEvaluator::reduceAt(const Dimension& dim) {
    innerLoopsShape.emplace_back(dim.size());
    innerLoops.emplace_back(VariableValueNode::Create(true, innerLoops.size(), "ri_" + std::to_string(innerLoops.size())));
    assign(dim, innerLoops.back());
}

void DimensionEvaluator::fillWithReductions() {
    while (!remaining.empty()) {
        Dimension remainder = *remaining.begin();
        KAS_ASSERT(!values[remainder]);
        bfsQueue.emplace(remainder, Direction::Down);
        bfsQueue.emplace(remainder, Direction::Up);
        bfsVisited.emplace(remainder, 0);
        highestPriority = 0;
        bestCandidate = remainder;
        while (!bfsQueue.empty()) {
            auto [dim, direction] = bfsQueue.front();
            bfsQueue.pop();
            switch (direction) {
            case Direction::Down:
                walkDown<false>(dim);
                break;
            case Direction::Up:
                walkUp<false>(dim);
                break;
            }
        }
        reduceAt(*bestCandidate);

        // Clean up for next iteration.
        bfsQueue = decltype(bfsQueue) {};
        bfsVisited.clear();
        highestPriority = std::numeric_limits<int>::min();
        bestCandidate = std::nullopt;
    }
}

std::vector<IteratorValue> DimensionEvaluator::extractValues(const std::vector<Dimension>& dims) const {
    KAS_ASSERT(remaining.empty());
    std::vector<IteratorValue> result;
    std::ranges::move(dims | std::views::transform([&](const auto& dim) { return values.at(dim); }), std::back_inserter(result));
    return result;
}

AbstractAccess DimensionEvaluator::toAccess(int position, const std::vector<PureTensor>& inputs, const std::vector<Dimension>& output) {
    std::vector<std::vector<IteratorValue>> inputsAccesses;
    std::ranges::move(inputs | std::views::transform([&](const auto& tensor) { return extractValues(tensor.getDimensions()); }), std::back_inserter(inputsAccesses));
    return AbstractAccess {
        .position = position,
        .outerLoops = std::move(outerLoops),
        .innerLoops = std::move(innerLoops),
        .innerLoopsShape = std::move(innerLoopsShape),
        .inputs = std::move(inputsAccesses),
        .output = extractValues(output),
    };
}

} // namespace Derivative

} // namespace kas
