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
            return true;
        },
        [this](const SplitLikeVertex& s, auto from) {
            excludeUpwards(s.op.getInput());
            assertInsert(s[SplitLikeOp::OtherOutputBranch(from)]);
            return true;
        },
        [this](const MergeLikeVertex& m, auto) {
            excludeUpwards(m.op.getInputL());
            excludeUpwards(m.op.getInputR());
            return true;
        },
        [](const ExpandVertex& e, auto) {
            return true;
        },
    });
    KAS_ASSERT(success, "Encountered input in the propagation of wavefront.");
}

bool DependentCutSetDiscoverer::pushDownwards(const Dimension& dimension) {
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

DependentCutSetDiscoverer& DependentCutSetDiscoverer::include(const Dimension& dimension) {
    // First remove the dimension, and push the cut set downward.
    excludeUpwards(dimension);
    // Then compensate the cut set.
    assertInsert(dimension);
    return *this;
}

DependentCutSetDiscoverer& DependentCutSetDiscoverer::fill() {
    const auto dims = std::vector(cutSet.begin(), cutSet.end());
    for (const Dimension& dim: dims) {
        pushDownwards(dim);
    }
    return *this;
}

std::size_t DependentCutSetDiscoverer::removeReductions() {
    return std::erase_if(cutSet, [](const Dimension& dim) { return dim.is(DimensionType::Reduce); });
}

std::vector<Dimension> DependentCutSetDiscoverer::build() const {
    return std::vector<Dimension>(cutSet.begin(), cutSet.end());
}

Generator<RFactorSolver::Scheme> RFactorSolver::PlausibleRFactorSchemes(std::vector<const Reduce *> remaining, bool allowEmpty) {
    if (remaining.empty()) {
        KAS_ASSERT(allowEmpty);
        co_yield Scheme { {} };
        co_return;
    }
    KAS_ASSERT(remaining.size() <= std::numeric_limits<std::size_t>::digits);
    std::size_t mask = (static_cast<std::size_t>(1) << remaining.size()) - 1;
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

std::size_t IR::getFLOPs(const BindingContext& ctx) const {
    std::size_t flops = 0;
    forEach([&](const Tensor& tensor) {
        flops += tensor.getFLOPs(ctx);
    });
    return flops;
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

    return { std::move(inputs), std::move(current) };
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
    // TODO!!!
    co_return;
}

IR IRBuilder::build(const ContractionScheme& scheme, const BindingContext& ctx) const {
    auto ir = initial(scheme);
    rfactor(ir, ctx);
    optimizeLayout(ir);
    performViews(ir);
    return ir;
}

IR IRBuilder::Build(const std::vector<Topmost>& tensors, const BindingContext& ctx) {
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

} // namespace kas
