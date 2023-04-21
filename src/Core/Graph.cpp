#include "KAS/Core/Graph.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

Dimension RepeatLikeVertex::operator[](RepeatLikeOp::Branch branch) const {
    switch (branch) {
    case RepeatLikeOp::Branch::Input: return op.getInput();
    case RepeatLikeOp::Branch::Output: return op.output;
    }
    KAS_UNREACHABLE();
}
Direction RepeatLikeVertex::outgoingDirection(RepeatLikeOp::Branch branch) const {
    switch (branch) {
    case RepeatLikeOp::Branch::Input: return Direction::Up;
    case RepeatLikeOp::Branch::Output: return Direction::Down;
    }
    KAS_UNREACHABLE();
}
VisitedVertex RepeatLikeVertex::visitAdjacent(RepeatLikeOp::Branch branch) const {
    return graph.visitAlong(operator[](branch), outgoingDirection(branch));
}

Dimension SplitLikeVertex::operator[](SplitLikeOp::Branch branch) const {
    switch (branch) {
    case SplitLikeOp::Branch::Input: return op.getInput();
    case SplitLikeOp::Branch::OutputLhs: return op.outputLhs;
    case SplitLikeOp::Branch::OutputRhs: return op.outputRhs;
    }
    KAS_UNREACHABLE();
}
Direction SplitLikeVertex::outgoingDirection(SplitLikeOp::Branch branch) const {
    switch (branch) {
    case SplitLikeOp::Branch::Input: return Direction::Up;
    case SplitLikeOp::Branch::OutputLhs: return Direction::Down;
    case SplitLikeOp::Branch::OutputRhs: return Direction::Down;
    }
    KAS_UNREACHABLE();
}
VisitedVertex SplitLikeVertex::visitAdjacent(SplitLikeOp::Branch branch) const {
    return graph.visitAlong(operator[](branch), outgoingDirection(branch));
}

Dimension MergeLikeVertex::operator[](MergeLikeOp::Branch branch) const {
    switch (branch) {
    case MergeLikeOp::Branch::InputLhs: return op.getInputL();
    case MergeLikeOp::Branch::InputRhs: return op.getInputR();
    case MergeLikeOp::Branch::Output: return op.output;
    }
    KAS_UNREACHABLE();
}
Direction MergeLikeVertex::outgoingDirection(MergeLikeOp::Branch branch) const {
    switch (branch) {
    case MergeLikeOp::Branch::InputLhs: return Direction::Up;
    case MergeLikeOp::Branch::InputRhs: return Direction::Up;
    case MergeLikeOp::Branch::Output: return Direction::Down;
    }
    KAS_UNREACHABLE();
}
VisitedVertex MergeLikeVertex::visitAdjacent(MergeLikeOp::Branch branch) const {
    return graph.visitAlong(operator[](branch), outgoingDirection(branch));
}

void Graph::Builder::visit(const Iterator& dim) {
    outputIterators.push_back(&dim);
}
void Graph::Builder::visit(const MapReduceOp& dim) {
    mapReduceIterators.push_back(&dim);
}
void Graph::Builder::visit(const RepeatLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { op };
    visit(op->output);
}
void Graph::Builder::visit(const SplitLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { std::make_pair(op, Order::Left) };
    visit(op->outputLhs);
    parent = { std::make_pair(op, Order::Right) };
    visit(op->outputRhs);
}
void Graph::Builder::visit(const MergeLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { op };
    visit(op->output);
}
void Graph::Builder::visit(const Dimension& dim) {
    auto [it, inserted] = theOtherEndOfEdge.insert({dim, parent});
    if (inserted) {
        DimVisitor::visit(dim);
    }
}

void Graph::Builder::add(const Dimension& dim) {
    parent = { std::monostate{} };
    visit(dim);
}

Graph Graph::Builder::build() {
    std::ranges::sort(outputIterators, [](const Iterator *lhs, const Iterator *rhs) {
        return lhs->getIndex() < rhs->getIndex();
    });
    std::ranges::sort(mapReduceIterators, [](const MapReduceOp *lhs, const MapReduceOp *rhs) {
        return lhs->getPriority() < rhs->getPriority();
    });
    return {
        std::move(theOtherEndOfEdge),
        std::move(outputIterators),
        std::move(mapReduceIterators)
    };
}

void Graph::WalkDownVisitor::visit(const RepeatLikeOp::Input& dim) {
    auto op = dim.getOp();
    result.emplace(std::pair { RepeatLikeVertex { graph, *op }, RepeatLikeOp::Branch::Input });
}
void Graph::WalkDownVisitor::visit(const SplitLikeOp::Input& dim) {
    auto op = dim.getOp();
    result.emplace(std::pair { SplitLikeVertex { graph, *op }, SplitLikeOp::Branch::Input });
}
void Graph::WalkDownVisitor::visit(const MergeLikeOp::Input& dim) {
    auto op = dim.getOp();
    Order order = dim.getOrder();
    result.emplace(std::pair { MergeLikeVertex { graph, *op }, MergeLikeOp::InputBranchFromOrder(order) });
}

void Graph::WalkUpVisitor::operator()(const RepeatLikeOp *op) {
    result.emplace(std::pair { RepeatLikeVertex { graph, *op }, RepeatLikeOp::Branch::Output });
}
void Graph::WalkUpVisitor::operator()(std::pair<const SplitLikeOp *, Order> opAndOrder) {
    auto& [op, order] = opAndOrder;
    result.emplace(std::pair { SplitLikeVertex { graph, *op }, SplitLikeOp::OutputBranchFromOrder(order) });
}
void Graph::WalkUpVisitor::operator()(const MergeLikeOp *op) {
    result.emplace(std::pair { MergeLikeVertex { graph, *op }, MergeLikeOp::Branch::Output });
}

VisitedVertex Graph::visitAlong(const Dimension& dim, Direction dir) const {
    switch (dir) {
        case Direction::Down: {
            WalkDownVisitor visitor { *this };
            visitor.visit(dim);
            return std::move(visitor.result);
        }
        case Direction::Up: {
            WalkUpVisitor visitor { *this };
            theOtherEndOfEdge.at(dim).visit(visitor);
            return std::move(visitor.result);
        }
    }
}

} // namespace kas
