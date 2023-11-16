#include "KAS/Transforms/Canonicalization.hpp"
#include "KAS/Transforms/Transforms.hpp"


namespace kas {

UnorderednessCanonicalizer::UnorderednessCanonicalizer(const Graph::DimensionMap<std::size_t>& unorderedDims):
    unorderedDims(unorderedDims) {}
auto UnorderednessCanonicalizer::transformInput(const Dimension& dim) -> Unorderedness {
    if (dim.is(DimensionTypeWithOrder::ShareR)) {
        // Weights are unordered.
        return { .isUnordered = true };
    } else if (auto it = unorderedDims.find(dim); it != unorderedDims.end()) {
        return { .isUnordered = true, .source = it->second };
    } else {
        return { .isUnordered = false };
    }
}
auto UnorderednessCanonicalizer::transformExpand(const Dimension& dim) -> Unorderedness {
    return { .isUnordered = true };
}
auto UnorderednessCanonicalizer::transform(const RepeatLikeOp& op) -> Unorderedness {
    auto result = at(op.getInput());
    result.sourceSplitOp = nullptr;
    return result;
}
auto UnorderednessCanonicalizer::transform(const SplitLikeOp& op) -> std::pair<Unorderedness, Unorderedness> {
    auto result = at(op.getInput());
    if (result.isUnordered) {
        if (auto split = dynamic_cast<const SplitOp *>(&op); split) {
            if (!result.sourceSplitOp) result.sourceSplitOp = split; // Retain the topmost Split.
        }
    }
    return { result, result };
}
auto UnorderednessCanonicalizer::transform(const MergeLikeOp& op) -> Unorderedness {
    const auto& lhs = at(op.getInputL()), & rhs = at(op.getInputR());
    if (!(lhs.isUnordered && rhs.isUnordered)) {
        // Either one ordered -> ordered.
        return { .isUnordered = false };
    } else {
        // OK, both unordered.
        if (lhs.source.has_value() != rhs.source.has_value()) {
            // Keep the source.
            return { .isUnordered = true, .source = lhs.source.has_value() ? *lhs.source : *rhs.source };
        } else if (lhs.source == rhs.source) {
            if (lhs.sourceSplitOp && rhs.sourceSplitOp && lhs.sourceSplitOp == rhs.sourceSplitOp) {
                // Merged unordered split dims with same source.
                // Unordered dimensions need not be split then merged again.
                uncanonical = true;
            }
            // Both without source or with the same source.
            return { .isUnordered = true, .source = lhs.source };
        } else {
            // Only case: different sources.
            return { .isUnordered = false };
        }
    }
}

bool IsCanonicalGivenUnorderedness(const Graph& graph, const Graph::DimensionMap<std::size_t>& unorderedDims) {
    UnorderednessCanonicalizer canonicalizer { unorderedDims };
    graph.accept(canonicalizer);
    if (canonicalizer.uncanonical) return false;
    // Note that reductions can be viewed as merges as well.
    std::set<const SplitOp *> reductionSourceSplitOps;
    for (auto it: graph.getReduceIterators()) {
        auto src = canonicalizer.at(it).sourceSplitOp;
        if (src) {
            auto [_, unique] = reductionSourceSplitOps.insert(src);
            if (!unique) return false;
        }
    }
    return true;
}

auto PoolingDiscoverer::transformInput(const Dimension& dim) -> bool { return true; }
auto PoolingDiscoverer::transformExpand(const Dimension& dim) -> bool { return true; }
auto PoolingDiscoverer::transform(const RepeatLikeOp& op) -> bool { return at(op.getInput()); }
auto PoolingDiscoverer::transform(const SplitLikeOp& op) -> std::pair<bool, bool> {
    bool result = at(op.getInput());
    return { result, result };
}
auto PoolingDiscoverer::transform(const MergeLikeOp& op) -> bool {
    if (auto share = dynamic_cast<const ShareOp *>(&op); share) {
        return false;
    } else {
        return at(op.getInputL()) && at(op.getInputR());
    }
}

bool IsPoolingTooLarge(const Graph& graph, const BindingContext& ctx, std::size_t maxPoolingFactor) {
    PoolingDiscoverer pooling;
    graph.accept(pooling);
    auto poolingFactor = Size::Identity(ctx);
    for (const Reduce *r: graph.getReduceIterators()) {
        if (pooling.at(r)) {
            poolingFactor *= r->getBase().getDomain();
        }
    }
    return poolingFactor.upperBoundEst(ctx) > maxPoolingFactor;
}

} // namespace kas
