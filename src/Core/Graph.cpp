#include <limits>

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
void Graph::Builder::visit(const MapReduce& dim) {
    mapReduceIterators.insert(&dim);
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
    auto [it, inserted] = dimMeta.try_emplace(dim, parent, ancestor);
    if (!inserted) {
        // Visited before. Now propagate ancestor.
        it->second.ancestors.merges(ancestor);
    }
    // Since we need to propagate the ancestor all the way down, we always need to visit, no matter inserted or not.
    DimVisitor::visit(dim);
}

void Graph::Builder::addTopmost(const Dimension& dim) {
    auto index = CompactIndices::Single(topmost.size());
    topmost.emplace_back(dim);
    parent = { std::monostate{} };
    ancestor = index;
    visit(dim);
}

Graph Graph::Builder::build() {
    auto outputIterators = std::vector<const Iterator *>(this->outputIterators.begin(), this->outputIterators.end());
    auto mapReduceIterators = std::vector<const MapReduce *>(this->mapReduceIterators.begin(), this->mapReduceIterators.end());
    std::ranges::sort(outputIterators, [](const Iterator *lhs, const Iterator *rhs) {
        return lhs->getIndex() < rhs->getIndex();
    });
    std::ranges::sort(mapReduceIterators, [](const MapReduce *lhs, const MapReduce *rhs) {
        return lhs->getPriority() < rhs->getPriority();
    });
    return {
        std::move(topmost),
        std::move(dimMeta),
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
            dimMeta.at(dim).opAbove.visit(visitor);
            return std::move(visitor.result);
        }
    }
}

std::vector<Graph::ConnectedComponent> Graph::computeConnectedComponents() const {
    // First collect the bottom dimensions.
    std::vector<Dimension> outputDims;
    std::ranges::copy(outputIterators, std::back_inserter(outputDims));
    std::ranges::copy(mapReduceIterators, std::back_inserter(outputDims));

    // Build the edges.
    auto bottom2top = std::vector<CompactIndices>(); // The ancestors of each output Dimension.
    auto top2bottom = std::vector<CompactIndices>(topmost.size(), CompactIndices::None()); // The descendants of each input Dimension.
    for (std::size_t bottomId = 0; bottomId < outputDims.size(); ++bottomId) {
        const auto& bottom = outputDims[bottomId];
        const auto& ancestors =  dimMeta.at(bottom).ancestors;
        bottom2top.emplace_back(ancestors);
        ancestors.foreach([&](std::size_t topId) {
            top2bottom[topId].merges(CompactIndices::Single(bottomId));
        });
    }

    // Initialize labels.
    constexpr std::size_t Unvisited = std::numeric_limits<std::size_t>::max();
    auto bottom2component = std::vector<std::size_t>(outputDims.size(), Unvisited);
    auto top2component = std::vector<std::size_t>(topmost.size(), Unvisited);

    // Define DFS.
    auto dfsFromTop = [&](const auto& self, const auto& dfsFromBottom, std::size_t topId, std::size_t current) {
        if (top2component[topId] != Unvisited) {
            return;
        }
        top2component[topId] = current;
        top2bottom[topId].foreach([&](std::size_t bottomId) {
            dfsFromBottom(dfsFromBottom, self, bottomId, current);
        });
    };
    auto dfsFromBottom = [&](const auto& self, const auto& dfsFromTop, std::size_t bottomId, std::size_t current) {
        if (bottom2component[bottomId] != Unvisited) {
            return;
        }
        bottom2component[bottomId] = current;
        bottom2top[bottomId].foreach([&](std::size_t topId) {
            dfsFromTop(dfsFromTop, self, topId, current);
        });
    };

    // Do DFS.
    std::size_t componentId = 0;
    for (std::size_t topId = 0; topId < topmost.size(); ++topId) {
        if (top2component[topId] == Unvisited) {
            dfsFromTop(dfsFromTop, dfsFromBottom, topId, componentId++);
        }
    }

    // Collect components.
    auto components = std::vector<ConnectedComponent>(componentId);
    for (std::size_t bottomId = 0; bottomId < outputDims.size(); ++bottomId) {
        components[bottom2component[bottomId]].outputs.emplace_back(outputDims[bottomId]);
    }
    for (std::size_t topId = 0; topId < topmost.size(); ++topId) {
        components[top2component[topId]].inputs.emplace_back(topmost[topId]);
    }

    return components;
}

} // namespace kas
