#include <limits>

#include "KAS/Core/Expand.hpp"
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
    outputIterators.insert(&dim);
}
void Graph::Builder::visit(const Reduce& dim) {
    reduceIterators.insert(&dim);
    if (auto op = dynamic_cast<const PrimitiveOp *>(&dim)) {
        ops.emplace(op);
    } else {
        KAS_CRITICAL("Found naked Reduce (not wrapped in a ReduceOp) when building the graph!");
    }
}
void Graph::Builder::visit(const RepeatLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { op };
    match(op->output);
    ops.emplace(op);
}
void Graph::Builder::visit(const SplitLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { std::make_pair(op, Order::Left) };
    match(op->outputLhs);
    parent = { std::make_pair(op, Order::Right) };
    match(op->outputRhs);
    ops.emplace(op);
}
void Graph::Builder::visit(const MergeLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { op };
    match(op->output);
    ops.emplace(op);
}
void Graph::Builder::match(const Dimension& dim) {
    auto [it, inserted] = dimMeta.try_emplace(dim, parent, ancestor);
    if (!inserted) {
        // Visited before. Now propagate ancestor.
        it->second.ancestors.merges(ancestor);
    }
    // Since we need to propagate the ancestor all the way down, we always need to visit, no matter inserted or not.
    dim.accept(*this);
}

Graph::Builder& Graph::Builder::addDimension(const Dimension& dim) {
    topmost.getDimensions().emplace_back(dim);
    ancestor = CompactIndices::Single(topmost.getDimensions().size() + topmost.getExpansions().size());
    parent = { std::monostate{} };
    match(dim);
    return *this;
}
Graph::Builder& Graph::Builder::addExpansion(const Expand *exp) {
    topmost.getExpansions().emplace_back(exp);
    ancestor = CompactIndices::Single(topmost.getDimensions().size() + topmost.getExpansions().size());
    parent = { std::monostate{} };
    match(exp->output);
    if (auto op = dynamic_cast<const PrimitiveOp *>(exp)) {
        ops.emplace(op);
    } else {
        KAS_CRITICAL("Found naked Expand (not wrapped in an ExpandOp) when building the graph!");
    }
    return *this;
}
Graph::Builder& Graph::Builder::addTopmost(const Topmost& interface) {
    addDimensions(interface.getDimensions());
    addExpansions(interface.getExpansions());
    return *this;
}

Graph Graph::Builder::build() {
    auto outputIterators = std::vector<const Iterator *>(this->outputIterators.begin(), this->outputIterators.end());
    auto reduceIterators = std::vector<const Reduce *>(this->reduceIterators.begin(), this->reduceIterators.end());
    std::ranges::sort(outputIterators, [](const Iterator *lhs, const Iterator *rhs) {
        return lhs->getIndex() < rhs->getIndex();
    });
    std::ranges::sort(reduceIterators, [](const Reduce *lhs, const Reduce *rhs) {
        return lhs->getPriority() < rhs->getPriority();
    });
    topmost.sort();
    return {
        std::move(topmost),
        std::move(dimMeta),
        std::move(outputIterators),
        std::move(reduceIterators),
        std::move(ops),
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
            dim.accept(visitor);
            return std::move(visitor.result);
        }
        case Direction::Up: {
            WalkUpVisitor visitor { *this };
            dimMeta.at(dim).opAbove.visit(visitor);
            return std::move(visitor.result);
        }
    }
}

const PrimitiveOp *Graph::getOpAbove(const Dimension& dim) const {
    struct Visitor {
        const PrimitiveOp *result = nullptr;
        void operator()(std::monostate) {
            result = nullptr;
        }
        void operator()(const RepeatLikeOp *op) {
            result = op;
        }
        void operator()(std::pair<const SplitLikeOp *, Order> opAndOrder) {
            result = opAndOrder.first;
        }
        void operator()(const MergeLikeOp *op) {
            result = op;
        }
    };
    Visitor v;
    dimMeta.at(dim).opAbove.visit(v);
    return v.result;
}

} // namespace kas
