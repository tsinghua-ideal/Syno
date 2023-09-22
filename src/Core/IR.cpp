#include "KAS/Core/Expand.hpp"
#include "KAS/Core/IR.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Utils/Ranges.hpp"


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
            excludeUpwards(r.op.getInput());
            excludeHook(&r.op);
            return true;
        },
        [this](const SplitLikeVertex& s, auto from) {
            excludeUpwards(s.op.getInput());
            assertInsert(s[SplitLikeOp::OtherOutputBranch(from)]);
            excludeHook(&s.op);
            return true;
        },
        [this](const MergeLikeVertex& m, auto) {
            excludeUpwards(m.op.getInputL());
            excludeUpwards(m.op.getInputR());
            excludeHook(&m.op);
            return true;
        },
        [](const ExpandVertex& e, auto) {
            // It seems we have not defined ExpandOp.
            // excludeHook(&e.op);
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

bool TensorContractor::pushDownwards(const Dimension& dimension) {
    return graph.visitAlong(dimension, Direction::Down).match(Match {
        [&](const RepeatLikeVertex& r, auto) {
            assertErase(dimension);
            pushDownwards(r.op.output);
            return true;
        },
        [&](const SplitLikeVertex& s, auto) {
            assertErase(dimension);
            pushDownwards(s.op.outputLhs);
            pushDownwards(s.op.outputRhs);
            return true;
        },
        [&](const MergeLikeVertex& m, auto) {
            KAS_ASSERT(!dimension.is(DimensionType::Share), "fill() is not allowed to fill ShareOp's! Maybe this contraction scheme is not valid?");
            if (tryErase(dimension.as<MergeLikeOp::Input>().getOther())) {
                assertErase(dimension);
                pushDownwards(m.op.output);
                return true;
            }
            return false;
        },
        [&](const ExpandVertex& e, auto) -> bool {
            KAS_UNREACHABLE();
        },
    });
}

Graph::CompactIndices TensorContractor::add(const std::vector<Dimension>& tensorOutput) {
    Graph::CompactIndices features = graph.getAncestors(tensorOutput);
    const auto inserted = includeUnchecked(tensorOutput);
    KAS_ASSERT(inserted == tensorOutput.size());
    return features;
}

void TensorContractor::performContractions(Graph::CompactIndices targets) {
    KAS_ASSERT(collected.disjoint(targets));
    auto result = collected.merged(targets);
    for (const MergeLikeOp *op:
        graph.getOps()
        | std::views::filter([](const PrimitiveOp *op) { return op->getType() == DimensionType::Share; })
        | std::views::transform([](const PrimitiveOp *op) { return dynamic_cast<const MergeLikeOp *>(op); })
    ) {
        const Dimension& candidate = op->output;
        const auto feature = graph.getAncestors(candidate);
        if (feature.contains(collected) && feature != collected && result.contains(feature)) {
            // This is a contraction.
            allowedShareOps.emplace(op);
        }
    }
    while (!allowedShareOps.empty()) {
        const MergeLikeOp *op = *allowedShareOps.begin();
        allowedShareOps.erase(allowedShareOps.begin());
        include(op->output);
    }
}

void TensorContractor::excludeHook(const PrimitiveOp *op) {
    if (op->getType() == DimensionType::Share) {
        // For sure we are doing contraction here.
        auto erased = allowedShareOps.erase(dynamic_cast<const MergeLikeOp *>(op));
        KAS_ASSERT(erased == 1, "Unexpectedly performed contraction!");
    }
}

TensorContractor::TensorContractor(const Graph& graph): DependentCutSetDiscoverer(graph) {
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
    const auto dims = std::vector(cutSet.begin(), cutSet.end());
    for (const Dimension& dim: dims) {
        pushDownwards(dim);
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
        co_yield Scheme { {} };
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
            co_yield Scheme { std::move(current) };
        } else {
            for (Scheme scheme: PlausibleRFactorSchemes(std::move(next), false)) {
                scheme.cons(current);
                co_yield std::move(scheme);
            }
        }
    }
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

    for (Scheme scheme: PlausibleRFactorSchemes(std::move(remainingReductions), tensor.hasContraction())) {
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
        KAS_ASSERT(removedReductions == reductionGroup.size(), "The number of applied reductions ({}) is not equal to the size ({}) of the reduction group.", removedReductions, reductionGroup.size());

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
            // TODO!!! Tie breaker.
            KAS_WARNING("Equal FLOPs {} for two contraction schemes!", optimal);
        }
    }

    return current;
}

Graph IR::buildGraph() const {
    return Graph::Builder()
        .addDimensions(inputTensors | std::views::transform(&Tensor::output) | std::views::join)
        .addExpansions(expansions | std::views::join)
        .build();
}

std::size_t IR::getFLOPs(const BindingContext& ctx) const {
    std::size_t flops = 0;
    forEach([&](const Tensor& tensor) {
        flops += tensor.getFLOPs(ctx);
    });
    return flops;
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

    KAS_ASSERT(scheme.contractions.at(0).at(0) == 0, "The first contraction must include the input tensor as the first tensor.");
    TensorContractor contractor { graph };

    Tensor current;
    std::vector<Tensor> currentInputs;
    for (const auto& contraction: scheme.contractions) {
        KAS_ASSERT(!contraction.empty());

        // Collect all the required dimensions.
        for (std::size_t index: contraction) {
            const Tensor& input = inputs.at(index);
            currentInputs.emplace_back(input);
            contractor.contract(input.output());
        }

        // And apply reductions.
        contractor.reduce();
        current = TensorImpl::CreateView(currentInputs, Bottommost(contractor.build()));
        currentInputs = { current };

        // Remove all the reductions, so we can continue to the next contraction.
        contractor.removeReductions();
    }

    // If we have not reached the output, perform required views. Note that by this time we have done all reductions.
    bool allDimensionsCollected = DimensionSetEqual(contractor.build(), graph.getOutputIterators());
    if (
        !allDimensionsCollected ||
        // Special case where we do not even need to perform any contraction, reduction or view.
        (allDimensionsCollected && current.isInputTensor() && !std::ranges::equal(current.output(), graph.getOutputIterators()))
    ) {
        contractor.fill();
        auto finalOutput = contractor.build();
        KAS_ASSERT(DimensionSetEqual(finalOutput, graph.getOutputIterators()));
        current = TensorImpl::CreateView(currentInputs, std::move(finalOutput), std::vector<const Reduce *>{});
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
    ir.forEach([&](Tensor& tensor) {
        if (!tensor.hasReduction()) return;
        auto solver = RFactorSolver(tensor, graph, ctx);
        // TODO!!! Add overflow.
        auto optimal = solver.optimalRFactorScheme();
        solver.apply(optimal);
    });
}

void IRBuilder::optimizeLayout(IR& ir) const {
    if (!std::ranges::equal(ir.outputTensor.output(), graph.getOutputIterators())) {
        KAS_ASSERT(!ir.outputTensor.isInputTensor());
        std::vector<Dimension> expectedOutput;
        std::ranges::copy(graph.getOutputIterators(), std::back_inserter(expectedOutput));
        ir.outputTensor.adjustLayout(&expectedOutput, nullptr);
    }
    // TODO!!! Really optimize locality.
}

void IRBuilder::performViews(IR& ir) const {
    // TODO!!!
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

    bool earlyReduction = false;
    // TODO!!!!! find early reduction.

    std::vector<std::vector<bool>> laterThan(numTensors, std::vector<bool>(numTensors));
    // The first tensor must be the earliest, which is by default the case.
    // TODO!!!!! fill this out.
    for (const MergeLikeOp *op:
        graph.getOps()
        | std::views::filter([](const PrimitiveOp *op) { return op->getType() == DimensionType::Share; })
        | std::views::transform([](const PrimitiveOp *op) { return dynamic_cast<const MergeLikeOp *>(op); })
    ) {
        const Dimension lhs = op->getInputL(), rhs = op->getInputR();
        const auto lhsFeatures = graph.getAncestors(lhs), rhsFeatures = graph.getAncestors(rhs);

        // Required by canonicalization.
        KAS_ASSERT(lhsFeatures.contains(inputFeatures) && rhsFeatures.disjoint(inputFeatures));

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
        if (earlyReduction && !scheme.contractions.empty()) {
            auto& c = scheme.contractions.front();
            c.insert(c.begin(), 0);
        } else {
            scheme.contractions.insert(scheme.contractions.begin(), {0});
        }
        co_yield std::move(scheme);
    }

    co_return;
}

IR IRBuilder::build(const ContractionScheme& scheme, const BindingContext& ctx) const {
    auto ir = initial(scheme);
    rfactor(ir, ctx);
    optimizeLayout(ir);
    performViews(ir);
    return ir;
}

} // namespace kas
