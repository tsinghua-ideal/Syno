#include <limits>

#include "KAS/Core/Expand.hpp"
#include "KAS/Core/Graph.hpp"
#include "KAS/Utils/Common.hpp"

// This is highly dangerous!
// We only want to use the inheritance between Expand and ExpandOp.
// Do not use any other function!
#include "KAS/Transforms/Expand.hpp"


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
    assignHeight(&dim, 0);
}
void Graph::Builder::visit(const Reduce& dim) {
    reduceIterators.insert(&dim);
    // Do not put ReduceOp is ops.
    assignHeight(&dim, 1);
}
void Graph::Builder::visit(const RepeatLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { op };
    match(op->output);
    ops.emplace(op);
    assignHeight(&dim, acquireHeight(op->output) + 1);
}
void Graph::Builder::visit(const SplitLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { std::make_pair(op, Order::Left) };
    match(op->outputLhs);
    parent = { std::make_pair(op, Order::Right) };
    match(op->outputRhs);
    ops.emplace(op);
    assignHeight(&dim, std::max(acquireHeight(op->outputLhs), acquireHeight(op->outputRhs)) + 1);
}
void Graph::Builder::visit(const MergeLikeOp::Input& dim) {
    auto op = dim.getOp();
    parent = { op };
    match(op->output);
    ops.emplace(op);
    assignHeight(&dim, acquireHeight(op->output) + 1);
}
void Graph::Builder::match(const Dimension& dim) {
    auto [it, inserted] = dimMeta.try_emplace(dim, parent, ancestor, -1);
    if (!inserted) {
        // Visited before. Now propagate ancestor.
        it->second.ancestors.merges(ancestor);
    }
    // Since we need to propagate the ancestor all the way down, we always need to visit, no matter inserted or not.
    dim.accept(*this);
}

int Graph::Builder::acquireHeight(const Dimension& dim) const {
    int result = dimMeta.at(dim).height;
    KAS_ASSERT(result != -1);
    return result;
}

void Graph::Builder::assignHeight(const Dimension& dim, int desired) {
    int& height = dimMeta.at(dim).height;
    KAS_ASSERT(height == -1 || height == desired);
    height = desired;
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
    auto op = static_cast<const ExpandOp *>(exp);
    parent = { op };
    match(exp->output);
    ops.emplace(op);
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
        return Reduce::LexicographicalLEQ(*lhs, *rhs);
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
void Graph::WalkUpVisitor::operator()(const ExpandOp *op) {
    result.emplace(ExpandVertex { *op });
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
    return dimMeta.at(dim).opAbove.visit(Match {
        [](std::monostate) -> const PrimitiveOp * { return nullptr; },
        [](const PrimitiveOp *op) -> const PrimitiveOp * { return op; },
        [](const std::pair<const SplitLikeOp *, Order>& opAndOrder) -> const PrimitiveOp * { return opAndOrder.first; },
    });
}

Graph::CompactIndices Graph::getAncestors(const Dimension& dim) const {
    return dimMeta.at(dim).ancestors;
}

int Graph::getHeight(const Dimension& dim) const {
    int result = dimMeta.at(dim).height;
    KAS_ASSERT(result != -1);
    return result;
}

ConstrainedGraph ConstrainedGraph::Builder::build() {
    // Here, we need to go bottom-up, because we do not know the expansions.
    DimensionSet dimensions;
    std::set<const PrimitiveOp *> ops;
    std::size_t touchedTops = 0;
    std::size_t stuckSplitLikes = 0;
    auto collector = [&](const auto& self, const Dimension& dim) -> void {
        auto [_, inserted] = dimensions.emplace(dim);
        KAS_ASSERT(inserted);
        if (top && top->contains(dim)) {
            // Stop at upper boundary.
            ++touchedTops;
            return;
        }
        graph.visitAlong(dim, Direction::Up).match(Match {
            [&](const RepeatLikeVertex& r, auto) {
                auto [_, inserted] = ops.emplace(&r.op);
                KAS_ASSERT(inserted);
                self(self, r.op.getInput());
            },
            [&](const SplitLikeVertex& s, SplitLikeOp::Branch from) {
                auto other = s[SplitLikeOp::OtherOutputBranch(from)];
                if (dimensions.contains(other)) {
                    --stuckSplitLikes;
                    auto [_, inserted] = ops.emplace(&s.op);
                    KAS_ASSERT(inserted);
                    self(self, s[SplitLikeOp::Branch::Input]);
                } else {
                    ++stuckSplitLikes;
                }
            },
            [&](const MergeLikeVertex& m, auto) {
                auto [_, inserted] = ops.emplace(&m.op);
                KAS_ASSERT(inserted);
                self(self, m.op.getInputL());
                self(self, m.op.getInputR());
            },
            [&](const ExpandVertex& e, auto) {
                auto [_, inserted] = ops.emplace(&e.op);
                KAS_ASSERT(inserted);
            },
        });
    };
    if (bottom) {
        for (const Dimension& dim: *bottom) {
            collector(collector, dim);
        }
    } else {
        for (Dimension dim: graph.getOutputIterators()) {
            collector(collector, dim);
        }
        for (Dimension dim: graph.getReduceIterators()) {
            collector(collector, dim);
        }
    }
    KAS_ASSERT(stuckSplitLikes == 0, "There are {} stuck split-like dimensions when building ConstrainedGraph! Maybe some cut-set is actually not cut-set!", stuckSplitLikes);
    if (top) {
        KAS_ASSERT(touchedTops == top->size(), "There are {}, expected {}, touched top dimensions when building ConstrainedGraph! Maybe some cut-set is actually not cut-set!", touchedTops, top->size());
    }
    return {
        graph,
        std::move(dimensions),
        std::move(ops),
        std::move(top),
        std::move(bottom),
    };
}

ConstrainedGraph::VisitedSubgraphVertex ConstrainedGraph::visitAlong(const Dimension& dim, Direction dir) const {
    KAS_ASSERT(dimensions.contains(dim));
    if (dir == Direction::Up && top && top->contains(dim)) {
        return { std::pair { Direction::Up, dim } };
    } else if (dir == Direction::Down && bottom && bottom->contains(dim)) {
        return { std::pair { Direction::Down, dim } };
    } else {
        return { graph->visitAlong(dim, dir) };
    }
}

} // namespace kas
