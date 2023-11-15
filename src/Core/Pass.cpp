#include <list>

#include "KAS/Core/IR.hpp"
#include "KAS/Core/Pass.hpp"
#include "KAS/Utils/Ranges.hpp"

// This is highly dangerous!
// We only want to use the inheritance between Expand and ExpandOp.
// Do not use any other function!
#include "KAS/Transforms/Expand.hpp"


namespace kas {

void DependentCutSetDiscoverer::excludeUpwards(const Dimension& dimension) {
    if (tryErase(dimension)) {
        return;
    }
    // The basic idea is simple: consider the op above the dimension. Now remove the input edges and add the output edges.
    // After that, remove the desired dimension. This preserves the cut set properties, and allows use to remove the non-existent dimension.
    // This propagates up like a wavefront.
    // When the wavefront touches the existing cut set, stop.
    // And follow the trace back, to push the cut set downward.
    bool success = graph.visitAlong(dimension, Direction::Up).match(Match {
        [this](const RepeatLikeVertex& r, auto) {
            beforeExclusionHook(&r.op);
            excludeUpwards(r.op.getInput());
            afterExclusionHook(&r.op);
            return true;
        },
        [this](const SplitLikeVertex& s, auto from) {
            beforeExclusionHook(&s.op);
            excludeUpwards(s.op.getInput());
            assertInsert(s[SplitLikeOp::OtherOutputBranch(from)]);
            afterExclusionHook(&s.op);
            return true;
        },
        [this](const MergeLikeVertex& m, auto) {
            beforeExclusionHook(&m.op);
            excludeUpwards(m.op.getInputL());
            excludeUpwards(m.op.getInputR());
            afterExclusionHook(&m.op);
            return true;
        },
        [this](const ExpandVertex& e, auto) {
            beforeExclusionHook(&e.op);
            afterExclusionHook(&e.op);
            return true;
        },
    });
    KAS_ASSERT(success, "Encountered input in the propagation of wavefront.");
}

DependentCutSetDiscoverer& DependentCutSetDiscoverer::include(const Dimension& dimension) {
    // First remove the dimension, and push the cut set downward.
    excludeUpwards(dimension);
    // Then compensate the cut set.
    assertInsert(dimension);
    return *this;
}

std::size_t DependentCutSetDiscoverer::removeReductions() {
    return std::erase_if(cutSet, [](const Dimension& dim) { return dim.is(DimensionType::Reduce); });
}

void DependentCutSetDiscoverer::removeSingleReduction(const Reduce *reduction) {
    auto erased = cutSet.erase(reduction);
    KAS_ASSERT(erased == 1);
}

std::vector<Dimension> DependentCutSetDiscoverer::build() const {
    return std::vector<Dimension>(cutSet.begin(), cutSet.end());
}

Graph::CompactIndices TensorContractor::add(const std::vector<Dimension>& tensorOutput) {
    Graph::CompactIndices features = graph.getAncestors(tensorOutput);
    KAS_ASSERT(collected.disjoint(features));
    const auto inserted = includeUnchecked(tensorOutput);
    KAS_ASSERT(inserted == tensorOutput.size());
    return features;
}

void TensorContractor::performContractions(Graph::CompactIndices targets) {
    KAS_ASSERT(collected.disjoint(targets));
    auto result = collected.merged(targets);
    for (const MergeLikeOp *op: graph.getOpsOfType<MergeLikeOp>(DimensionType::Share)) {
        const Dimension& candidate = op->output;
        const auto feature = graph.getAncestors(candidate);
        if (
            result.contains(feature) && // after contraction, we should have included this dimension.
            feature.excluded(collected) && // before contraction, this should not reside in the collected set, i.e., be done by it alone.
            feature.excluded(targets) // before contraction, this should not reside in the targets set, i.e., be done by it alone.
        ) {
            // This is a contraction.
            allowedShareOps.emplace(op);
        }
    }
    while (!allowedShareOps.empty()) {
        const MergeLikeOp *op = *allowedShareOps.begin();
        include(op->output);
    }
    collected = result;
}

void TensorContractor::beforeExclusionHook(const PrimitiveOp *op) {
    if (op->getType() == DimensionType::Share) {
        // For sure we are doing contraction here.
        auto erased = allowedShareOps.erase(dynamic_cast<const MergeLikeOp *>(op));
        KAS_ASSERT(erased == 1, "Unexpectedly performed contraction!");
    }
}

TensorContractor::TensorContractor(const Graph& graph, const std::vector<Dimension>& current):
    DependentCutSetDiscoverer(graph),
    collected(Graph::CompactIndices::None())
{
    collected.merges(add(current));
    collected.merges(graph.getAncestors(
        graph.getTopmost().getExpansions()
        | std::views::transform(Expand::PointerToDimension{})
    ));
}

TensorContractor& TensorContractor::reduce() {
    for (const Reduce *reduction: graph.getReduceIterators()) {
        const Dimension reductionDim = reduction;
        const auto feature = graph.getAncestors(reductionDim);
        if (collected.contains(feature) && !doneReductions.contains(reduction)) {
            include(reductionDim);
        }
    }
    return *this;
}

TensorContractor& TensorContractor::fill() {
    for (Dimension dim: graph.getOutputIterators()) {
        include(dim);
    }
    for (const Reduce *reduction: graph.getReduceIterators()) {
        if (!doneReductions.contains(reduction)) {
            include(reduction);
        }
    }
    return *this;
}

TensorContractor& TensorContractor::removeReductions() {
    for (const Reduce *reduction:
        cutSet
        | std::views::filter([](const Dimension& dim) { return dim.is(DimensionType::Reduce); })
        | std::views::transform([](const Dimension& dim) { return &dim.as<Reduce>(); })
    ) {
        auto [_, inserted] = doneReductions.insert(reduction);
        KAS_ASSERT(inserted);
    }
    DependentCutSetDiscoverer::removeReductions();
    return *this;
}

Generator<RFactorSolver::Scheme> RFactorSolver::PlausibleRFactorSchemes(std::vector<const Reduce *> remaining, bool allowEmpty) {
    if (remaining.empty()) {
        KAS_ASSERT(allowEmpty);
        auto res = Scheme { {} };
        co_yield std::move(res);
        co_return;
    }
    KAS_ASSERT(remaining.size() <= std::numeric_limits<std::size_t>::digits);
    const std::size_t mask = (1_uz << remaining.size()) - 1;
    for (std::size_t i = !allowEmpty; i <= mask; ++i) {
        std::vector<const Reduce *> current, next;
        for (std::size_t j = 0; j < remaining.size(); ++j) {
            if ((i >> j) & 1) {
                current.emplace_back(remaining[j]);
            } else {
                next.emplace_back(remaining[j]);
            }
        }
        if (next.empty()) {
            auto res = Scheme { std::move(current) };
            co_yield std::move(res);
        } else {
            for (Scheme scheme: PlausibleRFactorSchemes(std::move(next), false)) {
                scheme.cons(current);
                co_yield std::move(scheme);
            }
        }
    }
    co_return;
}

Generator<RFactorSolver::Scheme> RFactorSolver::PlausibleSingleReductionRFactorSchemes(std::vector<const Reduce *> remaining, bool firstEmpty) {
    KAS_ASSERT(!remaining.empty() || firstEmpty);
    std::ranges::sort(remaining);
    do {
        auto res = Scheme(ranges::to<std::vector<std::vector<const Reduce *>>>(
            remaining
            | std::views::transform([](const Reduce *reduction) { return std::vector<const Reduce *>{reduction}; })
        ));
        if (firstEmpty) {
            res.cons({});
        }
        co_yield std::move(res);
    } while (std::ranges::next_permutation(remaining).found);
}

Generator<RFactorSolver::Scheme> RFactorSolver::plausibleRFactorSchemes() const {
    const auto contractedReductions = ranges::to<std::vector<const Reduce *>>(
        contractedInterface
        | std::views::transform([](const Dimension& dim) { return dim.tryAs<Reduce>(); })
        | std::views::filter([](const Reduce *reduction) { return reduction != nullptr; })
    );

    std::vector<const Reduce *> remainingReductions;
    {
        auto reductionsSet = std::set<const Reduce *>(tensor.reductions().begin(), tensor.reductions().end());
        for (const Reduce *reduction: contractedReductions) {
            auto erased = reductionsSet.erase(reduction);
            KAS_ASSERT(erased == 1);
        }
        remainingReductions = std::vector<const Reduce *>(reductionsSet.begin(), reductionsSet.end());
    }

    auto schemes = (singleReductionPerStage ? PlausibleSingleReductionRFactorSchemes : PlausibleRFactorSchemes)(
        std::move(remainingReductions),
        // Allow empty group if there is contraction, or there are contracted reductions.
        tensor.hasContraction() || !contractedReductions.empty()
    );

    for (Scheme scheme: schemes) {
        auto& contractedGroup = scheme.reductions.at(0);
        contractedGroup.insert(contractedGroup.end(), contractedReductions.begin(), contractedReductions.end());
        co_yield std::move(scheme);
    }
}

std::size_t RFactorSolver::getFLOPs(const Scheme& scheme, std::size_t overflow) const {
    auto discoverer = DependentCutSetDiscoverer(graph, contractedInterface);

    bool isContraction = tensor.hasContraction();
    std::size_t flops = 0;
    for (std::size_t reductionGroupIndex = 0; const auto& reductionGroup: scheme.reductions) {
        KAS_ASSERT(isContraction || !reductionGroup.empty(), "Only rfactor is allowed for non-contraction subgraphs.");

        // 1 tensor : 1 add -> 1 add.
        // 2 tensors: 1 add -> 1 fma.
        // 3 tensors: 1 add -> 1 mul + 1 fma.
        auto instsPerAddition = isContraction ? std::max<std::size_t>(tensor.inputs().size() - 1, 1) : 1;

        // Add dependencies required by the reductions.
        discoverer.include(reductionGroup);
        auto cutSet = discoverer.build();

        // numel is just the product of the cut set.
        auto numel = std::transform_reduce(
            cutSet.begin(), cutSet.end(),
            Size::Identity(ctx), std::multiplies<>(),
            [](const Dimension& dim) -> const Size& { return dim.size(); }
        );

        // flops += numel * instsPerAddition
        KAS_ASSERT(!ctx.getAllConsts().empty());
        flops += numel.evalSumAllConsts(ctx) * instsPerAddition;
        if (flops > overflow) {
            return Infinity;
        }

        // remove all reductions from discoverer
        if (singleReductionPerStage) {
            if (reductionGroup.size() == 1) {
                discoverer.removeSingleReduction(reductionGroup.front());
            } else {
                KAS_ASSERT(reductionGroupIndex == 0);
                auto removedReductions = discoverer.removeReductions();
                KAS_ASSERT(removedReductions == reductionGroup.size());
            }
        } else {
            auto removedReductions = discoverer.removeReductions();
            if (removedReductions != reductionGroup.size()) {
                // It seems there are some other reductions that had better get reduced at the same time.
                KAS_ASSERT(removedReductions > reductionGroup.size());
                // This is to say that this reduction scheme is actually invalid.
                // We only need to return infinity here.
                return Infinity;
            }
        }

        // Later on, only reductions.
        isContraction = false;
        ++reductionGroupIndex;
    }

    // After all these, we should obtain the output of the tensor.
    auto finalCutSet = discoverer.build();
    KAS_ASSERT(DimensionSetEqual(finalCutSet, tensor.output()));

    return flops;
}

RFactorSolver::RFactorSolver(Tensor& tensor, const Graph& graph, const BindingContext& ctx, bool singleReductionPerStage, std::size_t maxVRAM):
    tensor { tensor }, graph { graph }, ctx { ctx }, singleReductionPerStage { singleReductionPerStage }, maxVRAM { maxVRAM },
    contractedInterface(TensorContractor::Contract(graph, tensor.inputs() | std::views::transform(&Tensor::output)))
{
    KAS_ASSERT(tensor.hasReduction());
}

std::optional<RFactorSolver::Scheme> RFactorSolver::optimalRFactorScheme() const {
    auto schemes = plausibleRFactorSchemes();

    std::size_t overflow = Infinity;
    std::optional<Scheme> optimal;
    for (auto scheme: schemes) {
        auto flops = getFLOPs(scheme, overflow);
        if (flops < overflow) {
            overflow = flops;
            optimal = std::move(scheme);
        }
    }

    return optimal;
}

void RFactorSolver::apply(const Scheme& scheme) {
    KAS_ASSERT(tensor.hasReduction());
    auto discoverer = DependentCutSetDiscoverer(graph, contractedInterface);

    bool isFirst = true;
    Tensor current;
    for (const auto& reductionGroup: scheme.reductions) {
        KAS_ASSERT(tensor.hasContraction() || !reductionGroup.empty(), "Only rfactor is allowed for non-contraction subgraphs.");

        // Add dependencies required by the reductions.
        discoverer.include(reductionGroup);
        auto cutSet = discoverer.build();
        for (Dimension r: reductionGroup) {
            auto erased = std::erase(cutSet, r);
            KAS_ASSERT(erased == 1);
        }

        // Apply rfactor.
        if (isFirst) {
            current = TensorImpl::CreateView(tensor.inputs(), std::move(cutSet), reductionGroup);
        } else {
            current = TensorImpl::CreateView(std::vector{current}, std::move(cutSet), reductionGroup);
        }

        // remove all reductions from discoverer
        if (singleReductionPerStage) {
            if (reductionGroup.size() == 1) {
                discoverer.removeSingleReduction(reductionGroup.front());
            } else {
                KAS_ASSERT(isFirst);
                auto removedReductions = discoverer.removeReductions();
                KAS_ASSERT(removedReductions == reductionGroup.size());
            }
        } else {
            auto removedReductions = discoverer.removeReductions();
            KAS_ASSERT(removedReductions == reductionGroup.size(), "The number of applied reductions ({}) is not equal to the size ({}) of the reduction group.", removedReductions, reductionGroup.size());
        }

        isFirst = false;
    }

    // After all these, we should obtain the output of the tensor.
    auto finalCutSet = discoverer.build();
    KAS_ASSERT(DimensionSetEqual(finalCutSet, tensor.output()));
    // And we can safely substitute the contents.
    tensor.getInputs() = std::move(current.getInputs());
    tensor.getOutput() = std::move(current.getOutput());
    tensor.getReductions() = std::move(current.getReductions());
}

RFactorIRPass::RFactorIRPass(const BindingContext& ctx, const Graph& graph, bool singleReductionPerStage, std::size_t maxVRAM):
    ctx { ctx }, graph { graph }, singleReductionPerStage { singleReductionPerStage }, maxVRAM { maxVRAM } {}

void RFactorIRPass::operator()(IR& ir) const {
    ir.topBottomForEach([&](Tensor& tensor) {
        if (!tensor.hasReduction()) return;
        auto solver = RFactorSolver(tensor, graph, ctx, singleReductionPerStage, maxVRAM);
        // TODO: Add overflow.
        auto optimal = solver.optimalRFactorScheme();
        if (!optimal.has_value()) {
            KAS_WARNING("RFactor failed for {}! reductions = [{}]", tensor.toString(ctx), fmt::join(tensor.reductions() | std::views::transform([this](const Reduce *reduction) { return reduction->getBase().getDomain().toString(ctx); }), ", "));
        } else {
            solver.apply(*optimal);
        }
    });
}

void OptimizeLayoutIRPass::optimize(Tensor& tensor) const {
    // We use a simple algorithm here.
    // First ignore the effect of weights, i.e., tensor.inputs() | std::views::drop(1).
    // Because we have optimized the layout of weights beforehand in LocalityOptimizer::permuteWeightDimensions.
    // TODO: unify the LocalityOptimizer.
    // Then, we try to keep the layout of the output as close as the input as possible.
    // - RepeatLikeOp, same priority.
    // - SplitLikeOp, the priority splits into two branches. lhs takes the outer loop, and rhs takes the inner loop.
    // - MergeLikeOp, if both operands has priority, then choose the inner loop. Otherwise, choose the one with priority.
    struct Priority {
        std::list<int> *serialized;
        using Placeholder = std::list<int>::iterator;
        std::optional<Placeholder> value;
        static Priority Append(std::list<int>& serialized) {
            serialized.emplace_back(0);
            return { &serialized, --serialized.end() };
        }
        static Priority Empty(std::list<int>& serialized) {
            return { &serialized };
        }
        std::strong_ordering operator<=>(const Priority& rhs) const {
            const Priority& lhs = *this;
            if (!lhs.value.has_value()) {
                if (!rhs.value.has_value()) {
                    return std::strong_ordering::equal;
                } else {
                    return std::strong_ordering::greater;
                }
            } else if (!rhs.value.has_value()) {
                return std::strong_ordering::less;
            }
            Placeholder l = *lhs.value, r = *rhs.value;
            if (l == r) {
                return std::strong_ordering::equal;
            }
            while (++l != serialized->end()) {
                if (l == r) {
                    // If l is before r in the list, l is in outer loop.
                    return std::strong_ordering::greater;
                }
            }
            l = *lhs.value;
            // Sanity check.
            while (++r != serialized->end()) {
                if (l == r) {
                    // If r is before l in the list, r is in outer loop.
                    return std::strong_ordering::less;
                }
            }
            KAS_UNREACHABLE();
        }
        static std::pair<Priority, Priority> Split(Priority from) {
            auto serialized = from.serialized;
            if (from.value.has_value()) {
                auto it = *from.value;
                auto lhs = Priority { serialized, from.serialized->insert(it, *it) };
                return { std::move(lhs), std::move(from) };
            } else {
                // Consider the special cases where canonicalization is broken.
                // That is, we are performing SplitLikeOp on weights!
                KAS_WARNING("Canonicalization is broken! Performing SplitLikeOp on weights!");
                return { Empty(*serialized), Empty(*serialized) };
            }
        }
        static Priority Merge(Priority lhs, Priority rhs) {
            return std::min(lhs, rhs);
        }
    };
    struct Assigner {
        const Graph& graph;
        std::list<int>& serialized;
        Graph::DimensionMap<Priority> priorities;
        int all = 0;
        void set(const Dimension& dim, Priority priority) {
            auto [_, inserted] = priorities.try_emplace(dim, std::move(priority));
            KAS_ASSERT(inserted);
        }
        Priority get(const Dimension& dim) {
            if (auto it = priorities.find(dim); it != priorities.end()) {
                return it->second;
            }
            graph.visitAlong(dim, Direction::Up).match(Match {
                [&](const RepeatLikeVertex& r, auto) {
                    set(dim, get(r.op.getInput()));
                },
                [&](const SplitLikeVertex& s, auto from) {
                    auto [lhs, rhs] = Priority::Split(get(s.op.getInput()));
                    set(s.op.outputLhs, std::move(lhs));
                    set(s.op.outputRhs, std::move(rhs));
                },
                [&](const MergeLikeVertex& m, auto) {
                    set(dim, Priority::Merge(get(m.op.getInputL()), get(m.op.getInputR())));
                },
                [&](const ExpandVertex& e, auto) {
                    set(dim, Priority::Empty(serialized));
                },
            });
            return priorities.at(dim);
        }
        Assigner(const Graph& graph, const Tensor& tensor, std::list<int>& serialized):
            graph(graph), serialized(serialized)
        {
            for (
                const Dimension& dim:
                tensor.inputs().at(0).output()
            ) {
                set(dim, Priority::Append(serialized));
            }
            for (
                const Dimension& dim:
                tensor.inputs() | std::views::drop(1) | std::views::transform(&Tensor::output) | std::views::join
            ) {
                set(dim, Priority::Empty(serialized));
            }
        }
        void serialize() {
            for (int& p: serialized) {
                p = all++;
            }
        }
        int priority(const Priority& priority) {
            if (priority.value) {
                return *(*priority.value);
            }
            return all++;
        }
    };
    std::list<int> serialized;
    auto assigner = Assigner(graph, tensor, serialized);
    std::vector<Priority> outputPriorities, reductionPriorities;
    for (const Dimension& dim: tensor.output()) outputPriorities.emplace_back(assigner.get(dim));
    for (Dimension dim: tensor.reductions()) reductionPriorities.emplace_back(assigner.get(dim));
    assigner.serialize();
    std::vector<std::size_t> outputIndices(outputPriorities.size()), reductionIndices(reductionPriorities.size());
    std::iota(outputIndices.begin(), outputIndices.end(), 0);
    std::iota(reductionIndices.begin(), reductionIndices.end(), 0);
    std::vector<int> outputPrioritiesValues, reductionPrioritiesValues;
    for (const Priority& priority: outputPriorities) outputPrioritiesValues.emplace_back(assigner.priority(priority));
    for (const Priority& priority: reductionPriorities) reductionPrioritiesValues.emplace_back(assigner.priority(priority));
    std::sort(outputIndices.begin(), outputIndices.end(), [&](std::size_t lhs, std::size_t rhs) {
        return outputPrioritiesValues[lhs] < outputPrioritiesValues[rhs];
    });
    std::sort(reductionIndices.begin(), reductionIndices.end(), [&](std::size_t lhs, std::size_t rhs) {
        return reductionPrioritiesValues[lhs] < reductionPrioritiesValues[rhs];
    });
    std::vector<Dimension> output;
    std::vector<const Reduce *> reductions;
    for (std::size_t index: outputIndices) output.emplace_back(tensor.output()[index]);
    for (std::size_t index: reductionIndices) reductions.emplace_back(tensor.reductions()[index]);
    tensor.getOutput() = std::move(output);
    tensor.getReductions() = std::move(reductions);
}

OptimizeLayoutIRPass::OptimizeLayoutIRPass(const Graph& graph): graph { graph } {}

void OptimizeLayoutIRPass::operator()(IR& ir) const {
    if (!std::ranges::equal(ir.outputTensor.output(), graph.getOutputIterators())) {
        KAS_ASSERT(!ir.outputTensor.isInputTensor());
        std::vector<Dimension> expectedOutput;
        std::ranges::copy(graph.getOutputIterators(), std::back_inserter(expectedOutput));
        ir.outputTensor.adjustLayout(&expectedOutput, nullptr);
    }
    ir.topBottomForEach([&](Tensor& tensor) {
        if (!tensor.isInputTensor() && ir.outputTensor != tensor) {
            optimize(tensor);
        }
    });
}

} // namespace kas
