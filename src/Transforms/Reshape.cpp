#include "KAS/Transforms/Merge.hpp"
#include "KAS/Transforms/Reshape.hpp"
#include "KAS/Transforms/Share.hpp"
#include "KAS/Transforms/Split.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

void ReshapeBlockNeighbors::Multiple::add(const ReshapeBlockNeighbors& neighbors) {
    if (!std::holds_alternative<std::monostate>(neighbors.left)) lefts.insert(neighbors.left);
    if (!std::holds_alternative<std::monostate>(neighbors.right)) rights.insert(neighbors.right);
}

auto ReshapeBlockNeighbors::separatedBy(const MergeOp *separator) const -> std::pair<Self, Self> {
    KAS_ASSERT(separator);
    return {
        { left, separator },
        { separator, right },
    };
};

auto ReshapeBlockNeighbors::isAdjacentTo(const Self& rhs) const -> bool {
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

auto ReshapeBlockNeighbors::isAdjacentTo(const Multiple& multiple) const -> bool {
    return
        (!std::holds_alternative<std::monostate>(left) && multiple.rights.contains(left)) ||
        (!std::holds_alternative<std::monostate>(right) && multiple.lefts.contains(right));
}

auto ReshapeBlockNeighbors::combinedWith(const Self& rhs) const -> Self {
    KAS_ASSERT(!isAdjacentTo(rhs));
    return { left, rhs.right };
}

auto ReshapeCanonicalizer::transform(const Iterator& dim) const -> Adjacent {
    return {};
}

auto ReshapeCanonicalizer::transform(const Reduce& dim) const -> Adjacent {
    return { &dim, &dim };
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
        return { at(op.output), {} };
    } else {
        return {};
    }
}

} // namespace kas
