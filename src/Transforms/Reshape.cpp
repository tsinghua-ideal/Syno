#include "KAS/Transforms/Merge.hpp"
#include "KAS/Transforms/Reshape.hpp"
#include "KAS/Transforms/Split.hpp"


namespace kas {

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

auto ReshapeBlockNeighbors::combinedWith(const Self& rhs) const -> Self {
    KAS_ASSERT(!isAdjacentTo(rhs));
    return { left, rhs.right };
}

auto ReshapeCanonicalizer::transform(const Iterator&) const -> Adjacent {
    return {};
}

auto ReshapeCanonicalizer::transform(const Reduce& reduction) const -> Adjacent {
    return { &reduction, &reduction };
}

auto ReshapeCanonicalizer::transform(const RepeatLikeOp::Input&) const -> Adjacent {
    return {};
}

auto ReshapeCanonicalizer::transform(const SplitLikeOp::Input& dim) const -> Adjacent {
    if (auto split = dynamic_cast<const SplitOp::Input *>(&dim); split) {
        auto op = split->getOp();
        return at(op->outputLhs).combinedWith(at(op->outputRhs));
    } else {
        return {};
    }
}

auto ReshapeCanonicalizer::transform(const MergeLikeOp::Input& dim) const -> std::pair<Adjacent, Adjacent> {
    if (auto merge = dynamic_cast<const MergeOp::Input *>(&dim); merge) {
        auto op = merge->getDerivedOp<MergeOp>();
        return at(op->output).separatedBy(op);
    } else {
        return {};
    }
}

} // namespace kas
