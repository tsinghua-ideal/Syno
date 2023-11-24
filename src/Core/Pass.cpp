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

LayoutOptimizer::Priority LayoutOptimizer::Priority::Head(std::list<int>& serialized) {
    serialized.emplace_front(0);
    return { &serialized, serialized.begin() };
}
LayoutOptimizer::Priority LayoutOptimizer::Priority::Tail(std::list<int>& serialized) {
    serialized.emplace_back(0);
    return { &serialized, std::prev(serialized.end()) };
}
LayoutOptimizer::Priority LayoutOptimizer::Priority::Empty(std::list<int>& serialized) {
    return { &serialized };
}
LayoutOptimizer::Priority LayoutOptimizer::Priority::Append(Priority from) {
    auto serialized = from.serialized;
    if (from.determined()) {
        auto it = *from.value;
        return { serialized, from.serialized->insert(std::next(it), *it) };
    } else {
        return Empty(*serialized);
    }
}
LayoutOptimizer::Priority LayoutOptimizer::Priority::Prepend(Priority from) {
    auto serialized = from.serialized;
    if (from.determined()) {
        auto it = *from.value;
        return { serialized, from.serialized->insert(it, *it) };
    } else {
        return Empty(*serialized);
    }
}

std::strong_ordering LayoutOptimizer::Priority::operator<=>(const Priority& rhs) const {
    const Priority& lhs = *this;
    KAS_ASSERT(lhs.serialized == rhs.serialized);
    if (!lhs.determined()) {
        if (!rhs.determined()) {
            return std::strong_ordering::equal;
        } else {
            return std::strong_ordering::greater;
        }
    } else if (!rhs.determined()) {
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

LayoutOptimizer::Priority LayoutOptimizer::Priority::MonadicMax(Priority lhs, Priority rhs) {
    if (lhs.determined() && rhs.determined()) {
        return std::max(lhs, rhs);
    } else if (lhs.determined()) {
        return lhs;
    } else if (rhs.determined()) {
        return rhs;
    } else {
        return Empty(*lhs.serialized);
    }
}
LayoutOptimizer::Priority LayoutOptimizer::Priority::MonadicMin(Priority lhs, Priority rhs) {
    if (lhs.determined() && rhs.determined()) {
        return std::min(lhs, rhs);
    } else if (lhs.determined()) {
        return lhs;
    } else if (rhs.determined()) {
        return rhs;
    } else {
        return Empty(*lhs.serialized);
    }
}

LayoutOptimizer::BottomTopPass::BottomTopPass(const Graph& graph, std::list<int>& serialized): serialized { serialized } {
    for (Dimension output: graph.getOutputIterators()) {
        preset(output, Priority::Tail(serialized));
    }
    for (Dimension reduction: graph.getReduceIterators()) {
        preset(reduction, Priority::Empty(serialized));
    }
}

auto LayoutOptimizer::BottomTopPass::transform(const RepeatLikeOp& op) const -> Priority {
    return at(op.output);
}
auto LayoutOptimizer::BottomTopPass::transform(const SplitLikeOp& op) const -> Priority {
    if (op.getType() == DimensionType::Unfold) {
        return at(op.outputLhs);
    } else {
        return Priority::MonadicMax(at(op.outputLhs), at(op.outputRhs));
    }
}
auto LayoutOptimizer::BottomTopPass::transform(const MergeLikeOp& op) const -> std::pair<Priority, Priority> {
    auto rhs = at(op.output);
    auto lhs = Priority::Prepend(rhs);
    return { std::move(lhs), std::move(rhs) };
}

void LayoutOptimizer::handleInputTensorAndExpansions(const std::vector<Dimension>& inputTensor, Graph::DimensionMap<Priority>& bottomTop) {
    // First the input.
    auto last = Priority::Empty(serialized);
    for (const Dimension& input: inputTensor | std::views::reverse) {
        auto& priority = bottomTop.at(input);
        if (!priority.determined()) {
            if (last.determined()) {
                // Prepend.
                priority = Priority::Prepend(last);
            } else {
                // Insert to head.
                priority = Priority::Tail(serialized);
            }
        }
        last = priority;
    }
    // Then the expansions.
    for (const Expand *expansion: graph.getTopmost().getExpansions()) {
        auto& priority = bottomTop.at(expansion->output);
        if (!priority.determined()) {
            priority = Priority::Tail(serialized);
        }
    }
}

LayoutOptimizer::TopBottomPass::TopBottomPass(const Graph& graph, std::list<int>& serialized, const Graph::DimensionMap<Priority>& bottomTop, bool unfoldToLeft):
    serialized { serialized }, bottomTop { bottomTop }, unfoldToLeft { unfoldToLeft } {}
auto LayoutOptimizer::TopBottomPass::transformInput(const Dimension& dim) -> Priority {
    // We cannot make sure we have determined this by now.
    // Because weights depend on ShareOp's.
    return bottomTop.at(dim);
}
auto LayoutOptimizer::TopBottomPass::transformExpand(const Dimension& dim) -> Priority {
    auto result = bottomTop.at(dim);
    KAS_ASSERT(result.determined());
    return result;
}
auto LayoutOptimizer::TopBottomPass::transform(const RepeatLikeOp& op) -> Priority {
    return at(op.getInput());
}
auto LayoutOptimizer::TopBottomPass::transform(const SplitLikeOp& op) -> std::pair<Priority, Priority> {
    Dimension inputDim = op.getInput();
    auto value = at(inputDim);
    const auto& leftBottomTopResult = bottomTop.at(op.outputLhs);
    const auto& rightBottomTopResult = bottomTop.at(op.outputRhs);
    // Check if this has been determined.
    if (leftBottomTopResult.determined() && rightBottomTopResult.determined()) {
        return { leftBottomTopResult, rightBottomTopResult };
    }
    auto other = Priority::Prepend(value);
    if (op.getType() == DimensionType::Unfold && unfoldToLeft) {
        return { std::move(value), std::move(other) };
    } else {
        return { std::move(other), std::move(value) };
    }
}
auto LayoutOptimizer::TopBottomPass::transform(const MergeLikeOp& op) -> Priority {
    auto lhs = at(op.getInputL());
    auto& rhs = attributes.at(op.getInputR());
    if (lhs.determined() && rhs.determined()) {
        auto bottomTopResult = bottomTop.at(op.output);
        if (bottomTopResult.determined()) {
            return bottomTopResult;
        } else {
            auto result = Priority::MonadicMax(lhs, rhs);
            KAS_ASSERT(result.determined());
            return result;
        }
    } else if (lhs.determined() || rhs.determined()) {
        // This is only possible for a ShareOp.
        KAS_ASSERT(op.getType() == DimensionType::Share && !rhs.determined());
        // First assign.
        rhs = lhs;
        // Then return.
        return lhs;
    } else {
        KAS_UNREACHABLE();
    }
}

void LayoutOptimizer::serialize() {
    for (int& p: serialized) {
        p = all++;
    }
}

void LayoutOptimizer::permute(Tensor& tensor, const Graph::DimensionMap<Priority>& priorities) {
    auto output = tensor.getOutput();
    auto reductions = tensor.getReductions();
    auto extract = [&](const Dimension& dim) { return priorities.at(dim).extract(); };
    std::ranges::sort(output, std::less<>{}, extract);
    std::ranges::sort(reductions, std::less<>{}, extract);
    tensor.adjustLayout(&output, &reductions);
}

LayoutOptimizer::LayoutOptimizer(const Graph& graph, bool unfoldToLeft):
    graph { graph }, unfoldToLeft { unfoldToLeft } {}

void LayoutOptimizer::optimize(IR& ir) {
    BottomTopPass bottomTopPass { graph, serialized };
    graph.accept(bottomTopPass);
    auto& bottomTop = bottomTopPass.attributes;

    const Tensor& data = ir.inputTensors.at(0);
    handleInputTensorAndExpansions(data.output(), bottomTop);

    TopBottomPass topBottomPass { graph, serialized, bottomTop, unfoldToLeft };
    graph.accept(topBottomPass);
    auto& topBottom = topBottomPass.attributes;

    serialize();

    ir.bottomTopForEach([&](Tensor& tensor) {
        if (tensor != ir.outputTensor && tensor != data) {
            permute(tensor, topBottom);
        }
    });
}

OptimizeLayoutIRPass::OptimizeLayoutIRPass(const Graph& graph, bool unfoldToLeft):
    graph { graph }, unfoldToLeft { unfoldToLeft } {}

void OptimizeLayoutIRPass::operator()(IR& ir) const {
    LayoutOptimizer rewriter { graph, unfoldToLeft };
    rewriter.optimize(ir);
    if (!std::ranges::equal(ir.outputTensor.output(), graph.getOutputIterators())) {
        KAS_ASSERT(!ir.outputTensor.isInputTensor());
        std::vector<Dimension> expectedOutput;
        std::ranges::copy(graph.getOutputIterators(), std::back_inserter(expectedOutput));
        ir.outputTensor.adjustLayout(&expectedOutput, nullptr);
    }
}

} // namespace kas
