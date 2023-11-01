#include "KAS/Transforms/Canonicalization.hpp"
#include "KAS/Transforms/Transforms.hpp"


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
    struct Checker: OpVisitor {
        const Graph& graph;
        const UnorderednessCanonicalizer& canonicalizer;
        bool uncanonical = false;
        Checker(const Graph& graph, const UnorderednessCanonicalizer& canonicalizer):
            graph { graph }, canonicalizer { canonicalizer } {}

        const SplitOp *getSourceSplitOp(const Dimension& dim) {
            return graph.visitAlong(dim, Direction::Up).match(Match {
                [](const RepeatLikeVertex& r, auto) -> const SplitOp * {
                    return nullptr;
                },
                [this](const SplitLikeVertex& s, auto) -> const SplitOp * {
                    if (s.op.getType() == SplitOp::Type) {
                        return getSourceSplitOp(s.op.getInput());
                    } else {
                        return nullptr;
                    }
                },
                [](const MergeLikeVertex& m, auto) -> const SplitOp * {
                    return nullptr;
                },
                [&](const ExpandVertex& e, auto) -> const SplitOp * {
                    return nullptr;
                },
            });
        }

        void visit(const ExpandOp& op) {}
        void visit(const ReduceOp& op) { KAS_UNREACHABLE(); }
        void visit(const MergeOp& op) {
            if (canonicalizer.at(op.output).isUnordered) {
                auto srcLhs = getSourceSplitOp(op.getInputL());
                auto srcRhs = getSourceSplitOp(op.getInputR());
                if (srcLhs && srcRhs && srcLhs == srcRhs) {
                    // Merged unordered split dims with same source.
                    // Unordered dimensions need not be split then merged again.
                    uncanonical = true;
                }
            }
        }
        void visit(const ShareOp& op) {}
        void visit(const ShiftOp& op) {
            // The channels are unordered, so shifting is of no use.
            if (canonicalizer.at(op.getInput()).isUnordered) {
                uncanonical = true;
            }
        }
        void visit(const SplitOp& op) {
            // Enforce size ordering of Split block.
            if (canonicalizer.at(op.getInput()).isUnordered) {
                const Size& lhs = op.outputLhs.size();
                const Size *rhs = nullptr;
                Dimension rhsDim = op.outputRhs;
                if (auto split = rhsDim.tryAs<SplitOp::Input>(); split) {
                    // If this is a split tower, the rhs size is the lhs of the child.
                    rhs = &split->getDerivedOp<SplitOp>()->getGroups();
                } else {
                    rhs = &rhsDim.size();
                }
                if (!Size::LexicographicalLEQ(lhs, *rhs)) {
                    uncanonical = true;
                }
            }
        }
        void visit(const StrideOp& op) {}
        void visit(const UnfoldOp& op) {
            // The channels are unordered, so there is no locality.
            if (canonicalizer.at(op.getInput()).isUnordered) {
                uncanonical = true;
            }
        }
    };
    Checker v { graph, canonicalizer };
    for (auto op: graph.getOps()) {
        op->accept(v);
        if (v.uncanonical) return false;
    }
    // Note that reductions can be viewed as merges as well.
    std::set<const SplitOp *> reductionSourceSplitOps;
    for (auto it: graph.getReduceIterators()) {
        auto src = v.getSourceSplitOp(it);
        if (src) {
            auto [_, unique] = reductionSourceSplitOps.insert(src);
            if (!unique) return false;
        }
    }
    return true;
}

} // namespace kas
