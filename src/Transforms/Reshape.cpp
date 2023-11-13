#include "KAS/Transforms/Merge.hpp"
#include "KAS/Transforms/Reshape.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

bool ReshapeBlockNeighbors::Sides::isAdjacentTo(const Sides& rhs) const {
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

bool ReshapeBlockNeighbors::Sides::isAdjacentTo(const SidesSet& multiple) const {
    return
        (!std::holds_alternative<std::monostate>(left) && multiple.rights.contains(left)) ||
        (!std::holds_alternative<std::monostate>(right) && multiple.lefts.contains(right));
}

auto ReshapeBlockNeighbors::Sides::separatedBy(const MergeOp *separator) const -> std::pair<Sides, Sides> {
    KAS_ASSERT(separator);
    return {
        { left, separator },
        { separator, right },
    };
}

auto ReshapeBlockNeighbors::Sides::combinedWith(const Sides& rhs) const -> Sides {
    KAS_ASSERT(!isAdjacentTo(rhs));
    return { left, rhs.right };
}

bool ReshapeBlockNeighbors::ContractedSides::isAdjacentTo(const ContractedSides& rhs) const {
    return std::ranges::any_of(sides, [&rhs](const auto& contracted) {
        const auto& [weightId, s] = contracted;
        if (auto it = rhs.sides.find(weightId); it != rhs.sides.end()) {
            return s.isAdjacentTo(it->second);
        } else {
            return false;
        }
    });
}

bool ReshapeBlockNeighbors::ContractedSides::isAdjacentTo(const ContractedSidesSet& multiple) const {
    return std::ranges::any_of(sides, [&multiple](const auto& contracted) {
        const auto& [weightId, s] = contracted;
        if (auto it = multiple.sides.find(weightId); it != multiple.sides.end()) {
            return s.isAdjacentTo(it->second);
        } else {
            return false;
        }
    });
}

auto ReshapeBlockNeighbors::ContractedSides::separatedBy(const MergeOp *separator) const -> std::pair<ContractedSides, ContractedSides> {
    KAS_ASSERT(separator);
    ContractedSides lhs, rhs;
    for (const auto& [weightId, s]: sides) {
        auto [l, r] = s.separatedBy(separator);
        lhs.sides.try_emplace(weightId, std::move(l));
        rhs.sides.try_emplace(weightId, std::move(r));
    }
    return { std::move(lhs), std::move(rhs) };
}

auto ReshapeBlockNeighbors::ContractedSides::separatedBy(const ShareOp *separator) const -> std::pair<ContractedSides, ContractedSides> {
    auto result = sides;
    auto [newWeight, created] = result.try_emplace(separator->getRhsOrigin());
    KAS_ASSERT(created);
    if (auto it = result.find(0); it != result.end()) {
        newWeight->second = std::move(it->second);
        result[0] = Sides{};
    }
    return { { std::move(result) }, {} };
}

auto ReshapeBlockNeighbors::ContractedSides::combinedWith(const ContractedSides& rhs) const -> ContractedSides {
    ContractedSides result;
    for (const auto& [weightId, s]: sides) {
        if (auto r = rhs.sides.find(weightId); r != rhs.sides.end()) {
            result.sides.try_emplace(weightId, s.combinedWith(r->second));
        } else {
            result.sides.try_emplace(weightId, s.combinedWith({}));
        }
    }
    for (const auto& [weightId, s]: rhs.sides) {
        if (auto r = sides.find(weightId); r == sides.end()) {
            result.sides.try_emplace(weightId, Sides{}.combinedWith(s));
        }
    }
    return result;
}

bool ReshapeBlockNeighbors::SidesSet::hasAdjacent() const {
    std::vector<Side> result;
    std::ranges::set_intersection(lefts, rights, std::back_inserter(result));
    return !result.empty();
}

bool ReshapeBlockNeighbors::ContractedSidesSet::hasAdjacent() const {
    return std::ranges::any_of(sides | std::views::values, &SidesSet::hasAdjacent);
}

void ReshapeBlockNeighbors::ContractedSidesSet::add(const ContractedSides& neighbors) {
    for (const auto& [weightId, s]: neighbors.sides) {
        auto& [lefts, rights] = sides[weightId];
        if (!std::holds_alternative<std::monostate>(s.left)) lefts.insert(s.left);
        if (!std::holds_alternative<std::monostate>(s.right)) rights.insert(s.right);
    }
}

void ReshapeBlockNeighbors::Multiple::add(const ReshapeBlockNeighbors& neighbors) {
    sidesSet.add(neighbors.sides);
}

bool ReshapeBlockNeighbors::Multiple::hasAdjacent() const {
    return sidesSet.hasAdjacent();
}

auto ReshapeBlockNeighbors::separatedBy(const MergeOp *separator) const -> std::pair<Self, Self> {
    auto [lhs, rhs] = sides.separatedBy(separator);
    return { { std::move(lhs) }, { std::move(rhs) } };
};

auto ReshapeBlockNeighbors::separatedBy(const ShareOp *separator) const -> std::pair<Self, Self> {
    auto [lhs, rhs] = sides.separatedBy(separator);
    return { { std::move(lhs) }, { std::move(rhs) } };
}

auto ReshapeBlockNeighbors::isAdjacentTo(const Self& rhs) const -> bool {
    return sides.isAdjacentTo(rhs.sides);
}

auto ReshapeBlockNeighbors::isAdjacentTo(const Multiple& multiple) const -> bool {
    return sides.isAdjacentTo(multiple.sidesSet);
}

auto ReshapeBlockNeighbors::combinedWith(const Self& rhs) const -> Self {
    return { sides.combinedWith(rhs.sides) };
}

auto ReshapeCanonicalizer::transform(const Iterator& dim) const -> Adjacent {
    return {};
}

auto ReshapeCanonicalizer::transform(const Reduce& dim) const -> Adjacent {
    Adjacent result;
    result.sides.sides.try_emplace(0, &dim, &dim);
    return result;
}

auto ReshapeCanonicalizer::transform(const RepeatLikeOp& op) const -> Adjacent {
    return {};
}

auto ReshapeCanonicalizer::transform(const SplitLikeOp& op) const -> Adjacent {
    if (auto split = dynamic_cast<const SplitOp *>(&op); split) {
        return at(op.outputLhs).combinedWith(at(op.outputRhs));
    } else {
        return {};
    }
}

auto ReshapeCanonicalizer::transform(const MergeLikeOp& op) const -> std::pair<Adjacent, Adjacent> {
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
