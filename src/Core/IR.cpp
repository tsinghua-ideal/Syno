#include <list>

#include "KAS/Core/Expand.hpp"
#include "KAS/Core/IR.hpp"
#include "KAS/Core/Dimension.hpp"
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
    const std::size_t mask = (static_cast<std::size_t>(1) << remaining.size()) - 1;
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

    for (Scheme scheme: PlausibleRFactorSchemes(
        std::move(remainingReductions),
        // Allow empty group if there is contraction, or there are contracted reductions.
        tensor.hasContraction() || !contractedReductions.empty()
    )) {
        auto& contractedGroup = scheme.reductions.at(0);
        contractedGroup.insert(contractedGroup.end(), contractedReductions.begin(), contractedReductions.end());
        co_yield std::move(scheme);
    }
}

std::size_t RFactorSolver::getFLOPs(const Scheme& scheme, std::size_t overflow) const {
    auto discoverer = DependentCutSetDiscoverer(graph, contractedInterface);

    bool isContraction = tensor.hasContraction();
    std::size_t flops = 0;
    for (const auto& reductionGroup: scheme.reductions) {
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
        for (const ConcreteConsts& consts: ctx.getAllConsts()) {
            flops += numel.eval<std::size_t>(consts) * instsPerAddition;
        }
        if (flops > overflow) {
            return Infinity;
        }

        // remove all reductions from discoverer
        auto removedReductions = discoverer.removeReductions();
        if (removedReductions != reductionGroup.size()) {
            // It seems there are some other reductions that had better get reduced at the same time.
            KAS_ASSERT(removedReductions > reductionGroup.size());
            // This is to say that this reduction scheme is actually invalid.
            // We only need to return infinity here.
            return Infinity;
        }

        // Later on, only reductions.
        isContraction = false;
    }

    // After all these, we should obtain the output of the tensor.
    auto finalCutSet = discoverer.build();
    KAS_ASSERT(DimensionSetEqual(finalCutSet, tensor.output()));

    return flops;
}

RFactorSolver::RFactorSolver(Tensor& tensor, const Graph& graph, const BindingContext& ctx):
    tensor { tensor }, graph { graph }, ctx { ctx },
    contractedInterface(TensorContractor::Contract(graph, tensor.inputs() | std::views::transform(&Tensor::output)))
{
    KAS_ASSERT(tensor.hasReduction());
}

std::optional<RFactorSolver::Scheme> RFactorSolver::optimalRFactorScheme(std::size_t overflow) const {
    auto schemes = plausibleRFactorSchemes();

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

        // Apply rfactor.
        if (isFirst) {
            current = TensorImpl::CreateView(tensor.inputs(), Bottommost(cutSet));
        } else {
            current = TensorImpl::CreateView(std::vector{current}, Bottommost(cutSet));
        }

        // remove all reductions from discoverer
        auto removedReductions = discoverer.removeReductions();
        KAS_ASSERT(removedReductions == reductionGroup.size(), "The number of applied reductions ({}) is not equal to the size ({}) of the reduction group.", removedReductions, reductionGroup.size());

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

IR IR::Build(const std::vector<Topmost>& tensors, const BindingContext& ctx) {
    auto builder = IRBuilder(tensors);
    auto schemes = builder.plausibleContractionSchemes();

    IR current;
    std::size_t optimal = std::numeric_limits<std::size_t>::max();
    for (auto scheme: schemes) {
        auto result = builder.build(scheme, ctx);
        auto flops = result.getFLOPs(ctx);
        if (flops < optimal) {
            current = std::move(result);
            optimal = flops;
        } else if (flops == optimal) {
            const auto currentStages = current.numStages(), resultStages = result.numStages();
            if (currentStages < resultStages) {
                // More stages enable schedulers to do more optimizations.
                current = std::move(result);
            } else if (currentStages == resultStages) {
                // TODO: Tie breaker.
                ++CountEqualFLOPs;
                // KAS_WARNING("Equal FLOPs {} and stages {} for two contraction schemes!", optimal, currentStages);
            }
        }
    }

    return current;
}

IR IR::copy() const {
    std::map<Tensor, Tensor> oldToNew;
    auto dfs = [&](const auto& self, const Tensor& tensor) -> void {
        if (oldToNew.contains(tensor)) return;
        for (const Tensor& input: tensor.inputs()) {
            self(self, input);
        }
        Tensor newTensor = tensor.clone(oldToNew);
        auto [_, inserted] = oldToNew.try_emplace(tensor, newTensor);
        KAS_ASSERT(inserted);
    };
    dfs(dfs, outputTensor);
    auto newInputs = ranges::to<std::vector<Tensor>>(
        inputTensors
        | std::views::transform([&](const Tensor& t) {
            return oldToNew.at(t);
        })
    );
    return { expansions, std::move(newInputs), oldToNew.at(outputTensor) };
}

Graph IR::buildGraph() const {
    return Graph::Builder()
        .addDimensions(inputTensors | std::views::transform(&Tensor::output) | std::views::join)
        .addExpansions(expansions | std::views::join)
        .build();
}

std::size_t IR::getFLOPs(const BindingContext& ctx, const ConcreteConsts& consts) const {
    std::size_t flops = 0;
    bottomTopForEach([&](const Tensor& tensor) {
        flops += tensor.getFLOPs(ctx, consts);
    });
    return flops;
}
std::size_t IR::getFLOPs(const BindingContext& ctx) const {
    std::size_t flops = 0;
    bottomTopForEach([&](const Tensor& tensor) {
        flops += tensor.getFLOPs(ctx);
    });
    return flops;
}

std::size_t IR::numStages() const {
    std::size_t stages = 0;
    bottomTopForEach([&](const Tensor& tensor) {
        stages += !tensor.isInputTensor();
    });
    return stages;
}

Generator<ContractionScheme> IRBuilder::plausibleContractionSchemes(const std::vector<std::vector<bool>>& laterThan, std::vector<std::size_t> remaining) const {
    if (remaining.empty()) {
        co_yield ContractionScheme { };
        co_return;
    }

    // If j is later than k (laterThan[j][k] == true), then k must be in a former or the same contraction group as that of j.
    const std::size_t mask = (static_cast<std::size_t>(1) << remaining.size()) - 1;
    for (std::size_t i = 1; i <= mask; ++i) {
        std::vector<std::size_t> contraction, next;
        for (std::size_t j = 0; j < remaining.size(); ++j) {
            if ((i >> j) & 1) {
                contraction.emplace_back(remaining[j]);
            } else {
                next.emplace_back(remaining[j]);
            }
        }
        bool valid = true;
        for (std::size_t j: contraction) {
            for (std::size_t k: next) {
                if (laterThan[j][k]) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }
        if (valid) {
            for (ContractionScheme scheme: plausibleContractionSchemes(laterThan, std::move(next))) {
                scheme.cons(contraction);
                co_yield std::move(scheme);
            }
        }
    }
}

IR IRBuilder::initial(const ContractionScheme& scheme) const {
    std::vector<Tensor> inputs;
    for (const Topmost& topmost: inputTensors) {
        inputs.emplace_back(TensorImpl::CreateInput(topmost.getDimensions()));
    }

    Tensor current = inputs.at(0);
    auto contractor = TensorContractor(graph, current.output());
    bool isFirst = true;
    for (const auto& contraction: scheme.contractions) {
        KAS_ASSERT(!contraction.empty() || isFirst, "Only when we do early reduction, we can have an empty contraction group.");

        std::vector<Tensor> currentInputs = { current };

        // Add newly contracted tensors to inputs.
        for (std::size_t index: contraction) {
            currentInputs.emplace_back(inputs.at(index));
        }
        // Collect all the required dimensions.
        contractor.contract(
            currentInputs
            | std::views::drop(1) // skip the current tensor.
            | std::views::transform(&Tensor::output)
        );

        // And apply reductions.
        contractor.reduce();
        current = TensorImpl::CreateView(currentInputs, Bottommost(contractor.build()));

        // Remove all the reductions, so we can continue to the next contraction.
        contractor.removeReductions();

        isFirst = false;
    }

    // If we have not reached the output, perform required views. Note that by this time we have done all reductions.
    bool allDimensionsCollected = DimensionSetEqual(contractor.build(), graph.getOutputIterators());
    if (
        !allDimensionsCollected ||
        // Special case where we do not even need to perform any contraction, reduction or view.
        (allDimensionsCollected && current.isInputTensor() && !std::ranges::equal(current.output(), graph.getOutputIterators()))
    ) {
        contractor.fill();
        auto finalOutput = Bottommost(contractor.build());
        KAS_ASSERT(DimensionSetEqual(finalOutput.getOutput(), graph.getOutputIterators()));
        current = TensorImpl::CreateView(std::vector<Tensor>{current}, std::move(finalOutput));
    }

    // We do not need to adjust the layout here, because rfactor pass will overwrite that anyway.
    // We adjust the output layout in layout pass.

    auto expansions = ranges::to<std::vector<std::vector<const Expand *>>>(
        inputTensors
        | std::views::transform(static_cast<const std::vector<const Expand *>& (Topmost::*)() const>(&Topmost::getExpansions))
    );
    return { std::move(expansions), std::move(inputs), std::move(current) };
}

void IRBuilder::rfactor(IR& ir, const BindingContext& ctx) const {
    ir.topBottomForEach([&](Tensor& tensor) {
        if (!tensor.hasReduction()) return;
        auto solver = RFactorSolver(tensor, graph, ctx);
        // TODO: Add overflow.
        auto optimal = solver.optimalRFactorScheme();
        if (!optimal.has_value()) {
            KAS_WARNING("RFactor failed for {}! reductions = [{}]", tensor.toString(ctx), fmt::join(tensor.reductions() | std::views::transform([&ctx](const Reduce *reduction) { return reduction->getBase().getDomain().toString(ctx); }), ", "));
        } else {
            solver.apply(*optimal);
        }
    });
}

void IRBuilder::optimizeLayout(IR& ir) const {
    if (!std::ranges::equal(ir.outputTensor.output(), graph.getOutputIterators())) {
        KAS_ASSERT(!ir.outputTensor.isInputTensor());
        std::vector<Dimension> expectedOutput;
        std::ranges::copy(graph.getOutputIterators(), std::back_inserter(expectedOutput));
        ir.outputTensor.adjustLayout(&expectedOutput, nullptr);
    }
    LayoutOptimizer().optimize(ir);
}

IRBuilder::IRBuilder(const std::vector<Topmost>& tensors):
    inputTensors { tensors },
    graph(Graph::Builder().addTopmosts(tensors).build())
{}

Generator<ContractionScheme> IRBuilder::plausibleContractionSchemes() const {
    const std::size_t numTensors = inputTensors.size();

    std::vector<Graph::CompactIndices> tensorFeatures;
    for (const auto& topmost: inputTensors) {
        tensorFeatures.emplace_back(graph.getAncestors(topmost.getAllDimensions()));
    }
    const auto& inputFeatures = tensorFeatures.at(0);

    // Create a dictionary.
    std::map<std::size_t, std::size_t> dimToOrigin;
    for (std::size_t i = 0; i < numTensors; ++i) {
        tensorFeatures[i].foreach([&](std::size_t index) {
            auto [it, inserted] = dimToOrigin.try_emplace(index, i);
            KAS_ASSERT(inserted || it->second == i);
        });
    }
    auto getOrigin = [&](const Dimension& dim) {
        std::set<std::size_t> origin;
        const auto dimFeatures = graph.getAncestors(dim);
        dimFeatures.foreach([&](std::size_t index) {
            origin.insert(dimToOrigin.at(index));
        });
        return origin;
    };

    bool earlyReduction = std::ranges::any_of(graph.getReduceIterators(), [&](const Reduce *reduction) {
        return inputFeatures.contains(graph.getAncestors(reduction));
    });

    std::vector<std::vector<bool>> laterThan(numTensors, std::vector<bool>(numTensors));
    // The first tensor must be the earliest, which is by default the case.
    for (const MergeLikeOp *op: graph.getOpsOfType<MergeLikeOp>(DimensionType::Share)) {
        const Dimension lhs = op->getInputL(), rhs = op->getInputR();
        const auto lhsFeatures = graph.getAncestors(lhs), rhsFeatures = graph.getAncestors(rhs);

        // Required by canonicalization.
        KAS_ASSERT(lhsFeatures.intersects(inputFeatures) && rhsFeatures.disjoint(inputFeatures));

        // First find the tensor that we are contracting.
        const auto rhsOriginSet = getOrigin(rhs);
        KAS_ASSERT(rhsOriginSet.size() == 1); // required by canonicalization.
        const auto rhsOrigin = *rhsOriginSet.begin();
        // Then the existing contracted tensors.
        const auto lhsOriginSet = getOrigin(lhs);
        for (std::size_t lhsOrigin: lhsOriginSet) {
            // Require that rhs be contracted later than lhs.
            laterThan[rhsOrigin][lhsOrigin] = true;
        }
    }

    for (ContractionScheme scheme:
        plausibleContractionSchemes(
            laterThan,
            ranges::to<std::vector<std::size_t>>(std::views::iota(static_cast<std::size_t>(1), numTensors))
        )
    ) {
        if (earlyReduction) {
            scheme.contractions.insert(scheme.contractions.begin(), std::vector<std::size_t>{});
        }
        co_yield std::move(scheme);
    }

    co_return;
}

IR IRBuilder::build(const ContractionScheme& scheme, const BindingContext& ctx) const {
    auto ir = initial(scheme);
    rfactor(ir, ctx);
    optimizeLayout(ir);
    return ir;
}

void LayoutOptimizer::optimize(const Graph& graph, Tensor& tensor) const {
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

void LayoutOptimizer::optimize(IR& ir) const {
    auto graph = ir.buildGraph();
    ir.topBottomForEach([&](Tensor& tensor) {
        if (!tensor.isInputTensor() && ir.outputTensor != tensor) {
            optimize(graph, tensor);
        }
    });
}

} // namespace kas
