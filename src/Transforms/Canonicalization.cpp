#include "KAS/Transforms/Canonicalization.hpp"
#include "KAS/Transforms/Shift.hpp"
#include "KAS/Transforms/Unfold.hpp"


namespace kas {

Unorderedness Unorderedness::Ordered() {
    return { false, std::nullopt };
}
Unorderedness Unorderedness::Unordered(std::size_t source) {
    return { true, source };
}
Unorderedness Unorderedness::Unordered() {
    return { true, std::nullopt };
}
Unorderedness Unorderedness::Unordered(std::optional<std::size_t> source) {
    return { true, std::move(source) };
}
Unorderedness Unorderedness::operator&&(const Unorderedness& rhs) const {
    if (!(isUnordered && rhs.isUnordered)) {
        // Either one ordered -> ordered.
        return Ordered();
    } else {
        // OK, both unordered.
        if (source.has_value() != rhs.source.has_value()) {
            // Keep the source.
            return Unordered(source.has_value() ? *source : *rhs.source);
        } else if (source == rhs.source) {
            // Both without source or with the same source.
            return Unordered(source);
        } else {
            // Only case: different sources.
            return Ordered();
        }
    }
}

UnorderednessCanonicalizer::UnorderednessCanonicalizer(const Graph::DimensionMap<std::size_t>& unorderedDims):
    unorderedDims(unorderedDims) {}
auto UnorderednessCanonicalizer::transformInput(const Dimension& dim) -> Unorderedness {
    if (dim.is(DimensionTypeWithOrder::ShareR)) {
        // Weights are unordered.
        return Unorderedness::Unordered();
    } else if (auto it = unorderedDims.find(dim); it != unorderedDims.end()) {
        return Unorderedness::Unordered(it->second);
    } else {
        return Unorderedness::Ordered();
    }
}
auto UnorderednessCanonicalizer::transformExpand(const Dimension& dim) -> Unorderedness {
    return Unorderedness::Unordered();
}
auto UnorderednessCanonicalizer::transform(const RepeatLikeOp& op) -> Unorderedness {
    return at(op.getInput());
}
auto UnorderednessCanonicalizer::transform(const SplitLikeOp& op) -> std::pair<Unorderedness, Unorderedness> {
    auto result = at(op.getInput());
    return { result, result };
}
auto UnorderednessCanonicalizer::transform(const MergeLikeOp& op) -> Unorderedness {
    return at(op.getInputL()) && at(op.getInputR());
}

bool IsCanonicalGivenUnorderedness(const Graph& graph, const Graph::DimensionMap<std::size_t>& unorderedDims) {
    UnorderednessCanonicalizer canonicalizer { unorderedDims };
    graph.accept(canonicalizer);
    for (auto shiftOp: graph.getOpsOfType<ShiftOp>()) {
        // The channels are unordered, so shifting is of no use.
        if (canonicalizer.at(shiftOp->getInput()).isUnordered) {
            return false;
        }
    }
    for (auto unfoldOp: graph.getOpsOfType<UnfoldOp>()) {
        // The channels are unordered, so there is no locality.
        if (canonicalizer.at(unfoldOp->getInput()).isUnordered) {
            return false;
        }
    }
    // TODO! Enforce size ordering of Split block.
    // TODO! Check merged unordered split dims with same source. Note that reductions can be viewed as merges as well.
    return true;
}

} // namespace kas
