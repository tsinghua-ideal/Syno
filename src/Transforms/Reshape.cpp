#include "KAS/Transforms/Merge.hpp"
#include "KAS/Transforms/Reshape.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

namespace Reshape {

bool Neighbors::isAdjacentTo(const Neighbors& rhs) const {
    return (
        // They originate from the very same MergeOp.
        std::holds_alternative<const MergeOp *>(right) && std::holds_alternative<const MergeOp *>(rhs.left)
        && std::get<const MergeOp *>(right) == std::get<const MergeOp *>(rhs.left)
    ) || (
        // Or they are commutative reductions. Actually we should check whether the reductions are commutative.
        // But we only have sum reduction for the time being.
        std::holds_alternative<const Reduce *>(right) && std::holds_alternative<const Reduce *>(rhs.left)
    );
}

bool Neighbors::isAdjacentTo(const NeighborsSet& multiple) const {
    return
        (!std::holds_alternative<std::monostate>(left) && multiple.rights.contains(left)) ||
        (!std::holds_alternative<std::monostate>(right) && multiple.lefts.contains(right));
}

std::pair<Neighbors, Neighbors> Neighbors::separatedBy(const MergeOp *separator) const {
    KAS_ASSERT(separator);
    return {
        { left, separator },
        { separator, right },
    };
}

Neighbors Neighbors::combinedWith(const Neighbors& rhs) const {
    KAS_ASSERT(!isAdjacentTo(rhs));
    return { left, rhs.right };
}

Block::Block():
    neighborsMap { { 0, {} } } {}

Block::Block(const Reduce& dim):
    neighborsMap { { 0, { &dim, &dim } } } {}

bool Block::isAdjacentTo(const Block& rhs) const {
    return std::ranges::any_of(neighborsMap, [&rhs](const auto& contracted) {
        const auto& [weightId, s] = contracted;
        if (auto it = rhs.neighborsMap.find(weightId); it != rhs.neighborsMap.end()) {
            return s.isAdjacentTo(it->second);
        } else {
            return false;
        }
    });
}

bool Block::isAdjacentTo(const BlockSet& multiple) const {
    return std::ranges::any_of(neighborsMap, [&multiple](const auto& contracted) {
        const auto& [weightId, s] = contracted;
        if (auto it = multiple.neighborsMap.find(weightId); it != multiple.neighborsMap.end()) {
            return s.isAdjacentTo(it->second);
        } else {
            return false;
        }
    });
}

std::pair<Block, Block> Block::separatedBy(const MergeOp *separator) const {
    KAS_ASSERT(separator);
    Block lhs, rhs;
    for (const auto& [weightId, s]: neighborsMap) {
        auto [l, r] = s.separatedBy(separator);
        lhs.neighborsMap.insert_or_assign(weightId, std::move(l));
        rhs.neighborsMap.insert_or_assign(weightId, std::move(r));
    }
    return { std::move(lhs), std::move(rhs) };
}

std::pair<Block, Block> Block::separatedBy(const ShareOp *separator) const {
    auto result = *this;
    auto [newWeight, created] = result.neighborsMap.try_emplace(separator->getRhsOrigin());
    KAS_ASSERT(created);
    newWeight->second = std::move(result.neighborsMap.at(0));
    result.neighborsMap[0] = {}; // Independently, brand new.
    return { std::move(result), {} };
}

Block Block::combinedWith(const Block& rhs) const {
    Block result;
    // Basically, merging two dicts.
    for (const auto& [weightId, s]: neighborsMap) {
        if (auto r = rhs.neighborsMap.find(weightId); r != rhs.neighborsMap.end()) {
            result.neighborsMap.insert_or_assign(weightId, s.combinedWith(r->second));
        } else {
            result.neighborsMap.insert_or_assign(weightId, s.combinedWith({}));
        }
    }
    for (const auto& [weightId, s]: rhs.neighborsMap) {
        if (auto r = neighborsMap.find(weightId); r == neighborsMap.end()) {
            result.neighborsMap.insert_or_assign(weightId, Neighbors{}.combinedWith(s));
        }
    }
    return result;
}

void BlockSet::add(const Block& block) {
    for (const auto& [weightId, s]: block.neighborsMap) {
        auto& [lefts, rights] = neighborsMap[weightId];
        if (!std::holds_alternative<std::monostate>(s.left)) lefts.insert(s.left);
        if (!std::holds_alternative<std::monostate>(s.right)) rights.insert(s.right);
    }
}

auto Block::AnyAdjacent(const std::vector<Block>& neighbors) -> bool {
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        for (std::size_t j = i + 1; j < neighbors.size(); ++j) {
            if (
                neighbors[i].isAdjacentTo(neighbors[j]) ||
                neighbors[j].isAdjacentTo(neighbors[i])
            ) {
                return true;
            }
        }
    }
    return false;
}

} // namespace Reshape

auto ReshapeCanonicalizer::transform(const Iterator& dim) const -> Reshape::Block {
    return {};
}

auto ReshapeCanonicalizer::transform(const Reduce& dim) const -> Reshape::Block {
    return Reshape::Block { dim };
}

auto ReshapeCanonicalizer::transform(const RepeatLikeOp& op) const -> Reshape::Block {
    return {};
}

auto ReshapeCanonicalizer::transform(const SplitLikeOp& op) const -> Reshape::Block {
    if (auto split = dynamic_cast<const SplitOp *>(&op); split) {
        return at(op.outputLhs).combinedWith(at(op.outputRhs));
    } else {
        return {};
    }
}

auto ReshapeCanonicalizer::transform(const MergeLikeOp& op) const -> std::pair<Reshape::Block, Reshape::Block> {
    if (auto merge = dynamic_cast<const MergeOp *>(&op); merge) {
        return at(op.output).separatedBy(merge);
    } else if (auto share = dynamic_cast<const ShareOp *>(&op); share) {
        // Peek over share.
        return at(op.output).separatedBy(share);
    } else {
        return {};
    }
}

} // namespace kas
