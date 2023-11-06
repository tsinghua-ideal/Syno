#include <fstream>
#include <ranges>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/CodeGen/PyTorchGen.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Transforms/Transforms.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

EinsumDiscoverer::EinsumDiscoverer(const ConstrainedGraph& subgraph, std::set<const ShareOp *>& remainingShares):
    DependentCutSetDiscoverer(subgraph.getGraph()),
    subgraph { subgraph },
    remainingShares(remainingShares)
{
    includeUnchecked(subgraph.getTop().value());
    includeUnchecked(
        subgraph.getOpsOfType<ExpandOp>()
        | std::views::transform(Expand::PointerToDimension{})
    );
}

Graph::DimensionSet EinsumDiscoverer::contract() {
    auto original = cutSet;

    bool contracted = false;
    while (true) {
        for (Dimension dim: cutSet) {
            if (auto shareInput = dim.tryAs<ShareOp::Input>(); shareInput) {
                Dimension other = shareInput->getOther();
                if (cutSet.contains(other)) {
                    contracted = true;
                    assertErase(dim);
                    assertErase(other);
                    auto op = shareInput->getDerivedOp<ShareOp>();
                    assertInsert(op->output);
                    auto erased = remainingShares.erase(op);
                    KAS_ASSERT(erased == 1);
                    break;
                }
            }
        }
        if (!contracted) break;
        contracted = false;
    }

    Graph::DimensionSet collectedShareBlocks;
    for (const Dimension& dim: cutSet) {
        if (!original.contains(dim)) {
            // If this is a new dimension, then it must
            collectedShareBlocks.insert(dim);
        }
    }
    return collectedShareBlocks;
}

EinsumContractor::EinsumContractor(const ConstrainedGraph& subgraph, std::set<const Reduce *>& remainingReductions):
    DependentCutSetDiscoverer(subgraph.getGraph()),
    subgraph { subgraph },
    remainingReductions(remainingReductions)
{
    includeUnchecked(subgraph.getTop().value());
    // We do not want to include ExpandOp's, because it is better to leave them to be lowered when needed.
}

EinsumContractor& EinsumContractor::contract() {
    bool contracted = false;
    while (true) {
        for (Dimension dim: cutSet) {
            // If there are contractions that can be done,
            // find the bottommost ShareOp.
            // Since we have processed the IR with PerformViewsIRPass, we are sure that there is only one ShareOp.
            while (
                dim.type() == DimensionType::Share &&
                !subgraph.getBottom().value().contains(dim)
            ) {
                contracted = true;
                dim = dim.as<ShareOp::Input>().getOp()->output;
            }
            if (contracted) {
                assignSubscript(dim, newSubscript());
                include(dim);
                if (auto reduction = dim.tryAs<Reduce>(); reduction) {
                    // Reduce that.
                    assertErase(dim);
                    auto erased = remainingReductions.erase(reduction);
                    KAS_ASSERT(erased);
                }
                break;
            }
        }
        if (!contracted) break;
        contracted = false;
    }
    for (const Dimension& dim: cutSet) {
        // If there are remaining dimensions, assign them.
        auto [_, inserted] = subscripts.try_emplace(dim, existingSubscripts);
        if (inserted) {
            ++existingSubscripts;
        }
    }
    return *this;
}

std::vector<Dimension> EinsumContractor::build(const std::vector<Tensor>& inputs) const {
    // TODO: we need to consider Iterator in weights.
    std::map<std::size_t, std::size_t> preferredOrder;
    std::size_t total = 0;
    for (const Dimension& dim: inputs | std::views::transform(&Tensor::output) | std::views::join) {
        auto [_, inserted] = preferredOrder.try_emplace(subscripts.at(dim), total);
        if (inserted) {
            ++total;
        }
    }
    auto result = DependentCutSetDiscoverer::build();
    std::ranges::sort(result, [&](const Dimension& lhs, const Dimension& rhs) {
        return preferredOrder.at(subscripts.at(lhs)) < preferredOrder.at(subscripts.at(rhs));
    });
    return result;
}

void EinsumContractor::beforeExclusionHook(const PrimitiveOp *op) {
    if (auto share = dynamic_cast<const ShareOp *>(op)) {
        // Assign subscripts.
        auto subscript = subscripts.at(share->output);
        assignSubscript(share->getInputL(), subscript);
        assignSubscript(share->getInputR(), subscript);
    } else {
        // Since we have transformed the IR into PyTorch form, we are sure that only ShareOp and ExpandOp are for contractions.
        KAS_ASSERT(op->getType() == DimensionType::Expand);
    }
}

std::vector<Dimension> PerformViewsIRPass::ViewPerformer::einsumContract() {
    // Prepare the remaining ShareOp's.
    std::ranges::copy(
        subgraph.getOpsOfType<ShareOp>(),
        std::inserter(remainingShares, remainingShares.begin())
    );
    // First find reachable share blocks.
    auto collectedShareBlocks =
        EinsumDiscoverer(subgraph, remainingShares)
        .contract();
    // Then perform them, by including the bottom most dim as dependency.
    return
        DependentCutSetDiscoverer(
            subgraph.getGraph(), subgraph.getTop().value()
        )
        .include(collectedShareBlocks)
        .build();
}

template<PerformViewsIRPass::ViewPerformer::State Marker, bool StopAtShare>
void PerformViewsIRPass::ViewPerformer::mark(const Dimension& dimension) {
    auto [_, attempt] = visited.try_emplace(dimension, Marker);
    if (!attempt) return;
    subgraph.visitAlong(dimension, Direction::Down).match(Match {
        [this](const RepeatLikeVertex& r, auto) {
            mark<Marker, StopAtShare>(r.op.output);
        },
        [this](const SplitLikeVertex& s, auto) {
            mark<Marker, StopAtShare>(s.op.outputLhs);
            mark<Marker, StopAtShare>(s.op.outputRhs);
        },
        [&](const MergeLikeVertex& m, auto) {
            if constexpr (StopAtShare) {
                if (!dimension.is(DimensionType::Share)) {
                    // Ignore dependencies
                    mark<Marker, StopAtShare>(m.op.output);
                }
            } else {
                mark<Marker, StopAtShare>(m.op.output);
            }
        },
        [&](const ExpandVertex& e, auto) {
            KAS_UNREACHABLE();
        },
        [&](Direction type, const Dimension& dim) {
            // Great. We have reached the bottom.
        },
    });
}
void PerformViewsIRPass::ViewPerformer::disable(const Dimension& dimension) {
    mark<State::Disabled, false>(dimension);
}
void PerformViewsIRPass::ViewPerformer::collect(const Dimension& dimension) {
    mark<State::Collected, true>(dimension);
}

std::vector<Dimension> PerformViewsIRPass::ViewPerformer::buildStage(const std::vector<Dimension>& cutSet) const {
    // We only need to find inputs of ShareOp's. They are all that are necessary.
    struct EinsumCutSetDiscoverer: public DependentCutSetDiscoverer {
        using DependentCutSetDiscoverer::DependentCutSetDiscoverer;
        void beforeExclusionHook(const PrimitiveOp *op) override {
            KAS_ASSERT(op->getType() != DimensionType::Share);
        }
    };
    auto discoverer = EinsumCutSetDiscoverer(subgraph.getGraph(), cutSet);
    for (const auto& [dim, state]: visited) {
        if (state == State::Collected && dim.is(DimensionType::Share)) {
            discoverer.include(dim);
        }
    }
    return discoverer.build();
}

PerformViewsIRPass::ViewPerformer::ViewPerformer(const Graph& graph, Tensor& tensor):
    tensor(tensor),
    subgraph(this->tensor.buildConstrainedGraph(graph))
{}

void PerformViewsIRPass::ViewPerformer::apply() {
    // We should first attempt to perform some views on the first input tensor.
    const auto& firstInput = tensor.inputs().at(0);
    // We only want to do views. So disable all weights.
    disable(tensor.inputs() | std::views::drop(1) | std::views::transform(&Tensor::output) | std::views::join);
    // Then collect the firstInput.
    collect(firstInput.output());
    // Great. Now let's see if we have performed any view.
    auto stage1 = buildStage(firstInput.output());
    if (!DimensionSetEqual(stage1, firstInput.output())) {
        // We have performed some views. Let's build a view.
        auto stage1Bottommost = Bottommost(stage1);
        auto viewTensor = TensorImpl::CreateView(
            std::vector<Tensor> { firstInput },
            stage1Bottommost
        );
        // Replace the input tensor.
        tensor.getInputs()[0] = viewTensor;
        auto& currentReductions = tensor.getReductions();
        auto [removeBegin, removeEnd] = std::ranges::remove_if(currentReductions, [&](const Reduce *reduction) {
            return std::ranges::find(stage1Bottommost.getReductions(), reduction) != stage1Bottommost.getReductions().end();
        });
        currentReductions.erase(removeBegin, removeEnd);
        // Now we have a new subgraph.
        subgraph = tensor.buildConstrainedGraph(subgraph.getGraph());
    }

    // Clear the marks so we can use `disable` and `collect` again.
    visited.clear();

    // Then we find out the einsum-contracted interface.
    auto einsumContractionResult = einsumContract();
    // If we have no remaining ShareOp's, then this Tensor is already able to be handled by PyTorchGen.
    // Skip.
    if (remainingShares.empty()) {
        return;
    } else if (warn) {
        // We need to perform additional Share.
        KAS_WARNING("This case is not lowerable to PyTorch. Taking diagonal entries!");
        if (warn > 5) {
            KAS_CRITICAL("Too many recursions.");
        }
    }

    // Then disable all descendants of ShareOp's.
    for (const ShareOp *share: remainingShares) {
        disable(share->output);
    }

    // Now that results of ShareOp's are disabled, all Dimension's left are reachable.
    // Collect them.
    for (const Dimension& dim: einsumContractionResult) {
        collect(dim);
    }
    // With the newly collected cut set as the stage tensor, build a view.
    auto stage2 = buildStage(einsumContractionResult);
    auto stage2Bottommost = Bottommost(stage2);
    // TODO: make this a helper function.
    auto viewTensor = TensorImpl::CreateView(
        tensor.inputs(),
        stage2Bottommost
    );
    // Replace the input tensor.
    tensor.getInputs() = { viewTensor };
    auto& currentReductions = tensor.getReductions();
    auto [removeBegin, removeEnd] = std::ranges::remove_if(currentReductions, [&](const Reduce *reduction) {
        return std::ranges::find(stage2Bottommost.getReductions(), reduction) != stage2Bottommost.getReductions().end();
    });
    currentReductions.erase(removeBegin, removeEnd);

    // Recursively perform another view.
    ViewPerformer(subgraph.getGraph(), tensor).shouldWarn(warn).apply();
}

PerformViewsIRPass::PerformViewsIRPass(const Graph& graph): graph(graph) {}

void PerformViewsIRPass::operator()(IR& ir) const {
    ir.topBottomForEach([&](Tensor& tensor) {
        if (tensor.hasContraction()) {
            ViewPerformer(graph, tensor).apply();
        }
    });
}

PyTorchGen::SubgraphGen::OpLower::OpLower(const BindingContext& ctx, const ConstrainedGraph& subgraph, PythonCodePrinter& printer, const ConcreteConsts& consts, std::vector<Dimension>& interface, const Tensor& tensor, const std::string& name):
    DependentCutSetDiscoverer(subgraph.getGraph(), interface),
    ctx { ctx }, subgraph { subgraph }, graph { subgraph.getGraph() }, printer { printer }, consts { consts }, interface { interface }, tensor { tensor }, name { name }
{}

void PyTorchGen::SubgraphGen::OpLower::reshapeToInterface() {
    printer.write("{0} = torch.reshape({0}, (", name);
    for (const Dimension& dim: interface) {
        printer.write("{}, ", concretize(dim.size()));
    }
    printer.writeLn("))");
}

std::array<std::size_t, 4> PyTorchGen::SubgraphGen::OpLower::reshapeToNCHW(std::size_t heightIndexInInterface, std::size_t heightSize) {
    std::size_t batchSize, channelSize;
    if (heightIndexInInterface == 0) {
        batchSize = 1;
        channelSize = 1;
    } else {
        batchSize = concretize(interface[0].size());
        channelSize = std::transform_reduce(interface.begin() + 1, interface.begin() + heightIndexInInterface, 1_uz, std::multiplies<std::size_t>(), [this](const Dimension& dim) {
            return concretize(dim.size());
        });
    }
    const std::size_t widthSize = std::transform_reduce(interface.begin() + heightIndexInInterface + 1, interface.end(), 1_uz, std::multiplies<std::size_t>(), [this](const Dimension& dim) {
        return concretize(dim.size());
    });
    printer.writeLn("{0} = torch.reshape({0}, ({1}, {2}, {3}, {4}, ))", name, batchSize, channelSize, heightSize, widthSize);

    return { batchSize, channelSize, heightSize, widthSize };
}

void PyTorchGen::SubgraphGen::OpLower::visit(const ExpandOp& op) {
    // We want to optimize for Repeat.
    if (
        auto merge = op.output.tryAs<MergeOp::Input>();
        // We have to make sure that this MergeOp is in this subgraph.
        merge && subgraph.getOps().contains(merge->getOp())
    ) {
        // This is a Repeat.
        undoneExpansions.try_emplace(op.output, &op);
        printer.writeLn("# Fused with {}", merge->getOp()->description(ctx));
    } else {
        // This is simple. Just unsqueeze and expand.
        printer.writeLn("{0} = torch.unsqueeze({0}, {1})", name, interface.size());
        interface.emplace_back(op.output);
        printer.write("{0} = {0}.expand(", name);
        for (const Dimension& dim: interface) {
            printer.write("{}, ", concretize(dim.size()));
        }
        printer.writeLn(")");
    }
}
void PyTorchGen::SubgraphGen::OpLower::visit(const Reduce& reduction) {
    printer.writeLn("# {}", reduction.description(ctx));
    auto inputIt = std::ranges::find(interface, &reduction);
    std::size_t inputIndex = std::distance(interface.begin(), inputIt);
    KAS_ASSERT(inputIndex < interface.size());
    printer.writeLn("{0} = torch.sum({0}, dim=({1}, ))", name, inputIndex);
    interface.erase(inputIt);
    printer.writeLn();
}
void PyTorchGen::SubgraphGen::OpLower::visit(const MergeOp& op) {
    const auto lhs = op.getInputL(), rhs = op.getInputR();
    std::size_t lhsIndex = std::distance(interface.begin(), std::ranges::find(interface, lhs));
    std::size_t rhsIndex = std::distance(interface.begin(), std::ranges::find(interface, rhs));
    const std::size_t length = interface.size();
    KAS_ASSERT(lhsIndex != rhsIndex);

    // Handle Repeat.
    if (lhsIndex == length || rhsIndex == length) {
        // This must be repeat.
        auto it = undoneExpansions.find(rhsIndex == length ? rhs : lhs);
        KAS_ASSERT(it != undoneExpansions.end());
        const ExpandOp *expandOp = it->second;
        undoneExpansions.erase(it);
        printer.writeLn("# Fused with {}", expandOp->description(ctx));
        printer.writeLn();
        visitRepeat(*expandOp, op);
        return;
    }

    // Modify the interface first.
    interface[std::max(lhsIndex, rhsIndex)] = op.output;
    interface.erase(interface.begin() + std::min(lhsIndex, rhsIndex));

    if (lhsIndex + 1 != rhsIndex) {
        // We need to permute.
        // Arrange the dimensions such that lhs and rhs are adjacent.
        const std::size_t objectiveRhs = std::max(lhsIndex, rhsIndex), beginInterval = std::min(lhsIndex, rhsIndex);
        const std::size_t objectiveLhs = objectiveRhs - 1;

        // Generate code.
        printer.write("{0} = torch.permute({0}, (", name);
        for (std::size_t i = 0; i < length; ++i) {
            std::size_t toWrite;
            if (i < beginInterval) {
                // No effect.
                toWrite = i;
            } else if (beginInterval <= i && i < objectiveLhs) {
                // Shift these dimensions to left.
                toWrite = i + 1;
            } else if (i == objectiveLhs) {
                toWrite = lhsIndex;
            } else if (i == objectiveRhs) {
                toWrite = rhsIndex;
            } else if (objectiveRhs < i) {
                // No effect.
                toWrite = i;
            } else {
                KAS_CRITICAL("Wrong index when lowering MergeOp.");
            }
            printer.write("{}, ", toWrite);
        }
        printer.writeLn("))");
        lhsIndex = objectiveLhs;
        rhsIndex = objectiveRhs;
    }

    // Now do the reshape.
    reshapeToInterface();
}
void PyTorchGen::SubgraphGen::OpLower::visit(const ShiftOp& op) {
    const auto [input, inputIndex] = getSingleInput(op);
    KAS_ASSERT(op.getShift() == 1, "Only shift by 1 is supported, because we alternate shift direction.");
    printer.writeLn("{0} = torch.roll({0}, self.shift_direction, {1})", name, inputIndex);
    interface[inputIndex] = op.output;
}
void PyTorchGen::SubgraphGen::OpLower::visit(const SplitOp& op) {
    const auto [input, inputIndex] = getSingleInput(op);

    // This is simple. A simple reshape will do it.
    interface[inputIndex] = op.outputRhs;
    interface.insert(interface.begin() + inputIndex, op.outputLhs);

    // Perform the reshape.
    reshapeToInterface();
}
void PyTorchGen::SubgraphGen::OpLower::visit(const StrideOp& op) {
    const auto [input, inputIndex] = getSingleInput(op);

    // Since interpolate only supports {3, 4, 5}-D tensors, we had better reshape the tensor to NCHW.
    const auto [batchSize, channelSize, heightSize, widthSize] = reshapeToNCHW(inputIndex, concretize(input.size()));
    const std::size_t stridedSize = concretize(op.output.size());
    printer.writeLn("{0} = torch.nn.functional.interpolate({0}, ({1}, {2}, ))", name, stridedSize, widthSize);

    // Then handle the interface.
    interface[inputIndex] = op.output;

    // Reshape it back.
    reshapeToInterface();
}
void PyTorchGen::SubgraphGen::OpLower::visit(const UnfoldOp& op) {
    const auto [input, inputIndex] = getSingleInput(op);

    // First pad the unfolded dimension.
    const std::size_t unpaddedHeightSize = concretize(input.size()), kernelSize = concretize(op.outputRhs.size());
    const std::size_t paddingLeft = kernelSize / 2, paddingRight = kernelSize - 1 - paddingLeft;
    bool useFusedPadding = paddingLeft == paddingRight; // If this is true, we can use the padding parameter in PyTorch.
    if (!useFusedPadding) {
        printer.write("{0} = torch.nn.functional.pad({0}, (", name);
        std::size_t numZeros = interface.size() - 1 - inputIndex;
        while (numZeros --> 0) {
            printer.write("0, 0, ");
        }
        printer.writeLn("{}, {}, ))", paddingLeft, paddingRight);
    }
    const std::size_t paddedHeightSize = useFusedPadding ? unpaddedHeightSize : unpaddedHeightSize + kernelSize - 1;

    // Since unfold in PyTorch only supports 4-D tensors, we have to reshape it.
    reshapeToNCHW(inputIndex, paddedHeightSize);

    // Then, unfold. First handle the interface.
    interface[inputIndex] = op.outputLhs;
    // PyTorch puts the window in channel dimension, that is, rhs is before lhs!
    interface.insert(interface.begin() + inputIndex, op.outputRhs);

    // Now apply the PyTorch unfold.
    if (useFusedPadding) {
        printer.writeLn("{0} = torch.nn.functional.unfold({0}, ({1}, 1, ), padding=({2}, 0, ))", name, kernelSize, paddingLeft);
    } else {
        printer.writeLn("{0} = torch.nn.functional.unfold({0}, ({1}, 1, ))", name, kernelSize);
    }

    // Finally, reshape it back.
    reshapeToInterface();
}
void PyTorchGen::SubgraphGen::OpLower::visitRepeat(const ExpandOp& expandOp, const MergeOp& mergeOp) {
    const auto op = RepeatOp { expandOp, mergeOp };
    printer.writeLn("# {}", op.description(ctx));
    const auto [input, inputIndex] = getSingleInput(op);

    const auto multiplier = concretize(op.getMultiplier());
    switch (op.getKind()) {
    case RepeatOp::Repeat:
        printer.writeLn("{0} = torch.repeat_interleave({0}, {1}, dim={2})", name, multiplier, inputIndex);
        break;
    case RepeatOp::Tile:
        printer.writeLn("{0} = torch.tile({0}, ({1}, ))", name, fmt::join(
            std::views::iota(0_uz, interface.size())
            | std::views::transform([&](std::size_t i) {
                if (i == inputIndex) return multiplier;
                else return 1_uz;
            }), ", "
        ));
        break;
    }

    interface[inputIndex] = op.output;
}

void PyTorchGen::SubgraphGen::OpLower::afterExclusionHook(const PrimitiveOp *op) {
    printer.writeLn("# {}", op->description(ctx));
    op->accept(*this);
    printer.writeLn();
}

void PyTorchGen::SubgraphGen::OpLower::checkDone() const {
    KAS_ASSERT(DimensionSetEqual(interface, tensor.output()));
    KAS_ASSERT(undoneExpansions.empty());
}

PyTorchGen::SubgraphGen::SubgraphGen(const BindingContext& ctx, const Graph& graph, const std::map<Tensor, std::string>& tensorNames, PythonCodePrinter& printer, const Tensor& tensor):
    ctx { ctx }, tensorNames(tensorNames), printer { printer }, tensor { tensor }, subgraph(tensor.buildConstrainedGraph(graph))
{
    remainingReductions.insert(tensor.reductions().begin(), tensor.reductions().end());
}

void PyTorchGen::SubgraphGen::generate(const ConcreteConsts& consts) {
    const auto& name = tensorNames.at(tensor);

    // Add Activation.
    if (false && tensor.inputs().size() > 1 && !tensor.inputs().at(0).isInputTensor()) {
        printer.writeLn("# Add activation before contraction.");
        printer.writeLn("{0} = torch.nn.functional.relu({0})", tensorNames.at(tensor.inputs()[0]));
        printer.writeLn();
    }

    // First perform contraction.
    auto contractor = EinsumContractor(subgraph, remainingReductions);
    auto interface = contractor.contract().build(tensor.inputs());
    const auto& subscripts = contractor.getSubscripts();
    auto dimToSubscript = [&](const Dimension& dim) -> std::size_t {
        return subscripts.at(dim);
    };
    std::vector<std::string> inputsSubscripts;
    for (const Tensor& inputTensor: tensor.inputs()) {
        inputsSubscripts.emplace_back(ToEinsteinNotation(inputTensor.output() | std::views::transform(dimToSubscript)));
    }
    printer.writeLn("# Perform contraction.");
    printer.writeLn(
        R"code({} = torch.einsum("{} -> {}", {}))code",
        name,
        fmt::join(inputsSubscripts, ", "),
        ToEinsteinNotation(interface | std::views::transform(dimToSubscript)),
        fmt::join(tensor.inputs() | std::views::transform([&](const Tensor& inputTensor) -> const std::string& {
            return tensorNames.at(inputTensor);
        }), ", ")
    );
    printer.writeLn();

    // Now we have got the input. Time to work on it. The objective is to reach `tensor.output()`.
    auto opLower = OpLower { ctx, subgraph, printer, consts, interface, tensor, name };
    for (const Reduce *reduction: remainingReductions) {
        opLower.include(reduction);
        // Reduce this.
        opLower.visit(*reduction);
    }
    // remainingReductions.clear(); // No need to clear it.
    // Then make sure all the output dims are visited as well.
    opLower.include(tensor.output());

    opLower.checkDone();

    // Well, we need to permute the interface so that it matches the output of this subgraph.
    // Maybe we can alter the output to get better performance? TODO
    std::vector<std::size_t> indices(interface.size(), std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 0; const Dimension& dim: interface) {
        auto index = std::distance(tensor.output().begin(), std::ranges::find(tensor.output(), dim));
        indices.at(index) = i;
        ++i;
    }
    if (std::ranges::adjacent_find(indices, [](std::size_t a, std::size_t b) { return a + 1 != b; }) != indices.end()) {
        // We need to permute.
        printer.writeLn("# Permute to match the output of this subgraph.");
        printer.write("{0} = torch.permute({0}, (", name);
        for (std::size_t i: indices) {
            printer.write("{}, ", i);
        }
        printer.writeLn("))");
        printer.writeLn();
    }
}

std::vector<std::size_t> PyTorchGen::concretize(const std::vector<Dimension>& interface, const ConcreteConsts& consts) const {
    return ShapeView(interface).eval<std::size_t>(consts);
}

PyTorchGen::PyTorchGen(const BindingContext& ctx, const IR& ir):
    ctx { ctx },
    ir { ir.copy() },
    graph { this->ir.buildGraph() }
{
    // We want rfactor to be applied so that each stage has at most 1 reduction.
    (RFactorIRPass(ctx, graph, true))(this->ir);
    // Perform views. Because PyTorch wants the dimensions to be contracted to be explicit.
    (PerformViewsIRPass(graph))(this->ir);
    // Now that the tensors are in a mess again, optimize layout one more time.
    (OptimizeLayoutIRPass(graph))(this->ir);

    for (std::size_t index = 0; const auto& tensor: this->ir.inputTensors) {
        declare(tensor, "in_" + std::to_string(index));
        ++index;
    }

    this->ir.topBottomForEach([&](const Tensor& tensor) {
        if (!tensor.isInputTensor()) {
            declare(tensor);
            topologicallyOrderedTensors.emplace_back(tensor);
        }
    });
}

void PyTorchGen::loadWeights(PythonCodePrinter& printer) const {
    for (std::size_t index = 1; const Tensor& weight: ir.inputTensors | std::views::drop(1)) {
        printer.writeLn("{} = self.weights[{}]", use(weight), index - 1);
        KAS_ASSERT(ir.expansions.at(index).empty(), "Expansion for weights unsupported.");
        ++index;
    }
}

void PyTorchGen::padInputTensor(PythonCodePrinter& printer, const PaddedConsts& consts) const {
    const Tensor& inputTensor = ir.inputTensors.at(0);
    const auto& inputShape = inputTensor.output();
    auto unpaddedShape = concretize(inputShape, consts.unpadded);
    auto paddedShape = concretize(inputShape, consts.padded);
    std::vector<std::pair<int, int>> paddingParams;
    for (std::size_t i = 0; i < inputShape.size(); ++i) {
        int delta = static_cast<int>(paddedShape[i]) - static_cast<int>(unpaddedShape[i]);
        KAS_ASSERT(delta >= 0, "Padded shape must be larger than unpadded shape!");
        if (delta == 0) {
            if (!paddingParams.empty()) {
                paddingParams.emplace_back(0, 0);
            }
        } else {
            paddingParams.emplace_back(delta / 2, delta - delta / 2);
        }
    }
    if (!paddingParams.empty()) {
        printer.writeLn("# We need to pad the input tensor.");
        printer.write("{} = torch.nn.functional.pad(x, (", use(inputTensor));
        for (const auto& [left, right]: paddingParams | std::views::reverse) {
            printer.write("{}, {}, ", left, right);
        }
        printer.writeLn("))");
    } else {
        printer.writeLn("# No need to pad the input tensor.");
        printer.writeLn("{} = x", use(inputTensor));
    }
    printer.writeLn();
}

void PyTorchGen::cropOutputTensor(PythonCodePrinter& printer, const PaddedConsts& consts) const {
    const auto& shape = ir.outputTensor.output();
    auto unpaddedShape = concretize(shape, consts.unpadded);
    auto paddedShape = concretize(shape, consts.padded);
    std::vector<std::pair<int, int>> slices;
    bool needsPad = false;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        int delta = static_cast<int>(paddedShape[i]) - static_cast<int>(unpaddedShape[i]);
        KAS_ASSERT(delta >= 0, "Padded shape must be larger than unpadded shape!");
        needsPad |= delta != 0;
        slices.emplace_back(delta / 2, delta / 2 - delta);
    }
    if (needsPad) {
        printer.writeLn("# We need to crop the output tensor.");
        printer.write("y = {}[", use(ir.outputTensor));
        for (const auto& [left, right]: slices) {
            if (left == 0 && right == 0) {
                printer.write(":, ");
            } else {
                printer.write("{}:{}, ", left, right);
            }
        }
        printer.writeLn("]");
    } else {
        printer.writeLn("# No need to crop the output tensor.");
        printer.writeLn("y = {}", use(ir.outputTensor));
    }
    printer.writeLn();
}

void PyTorchGen::applyDivision(PythonCodePrinter& printer, const ConcreteConsts& consts, const AbstractAccess& forwardAccess) const {
    const auto& divisor = forwardAccess.divBy;
    if (divisor) {
        printer.writeLn("y = y / {}", divisor->eval<std::size_t>(consts));
    }
}

void PyTorchGen::generatePrelude(std::ostream& outputStream) const {
    outputStream << "import torch\n";
    outputStream << "import random\n";
    outputStream << "\"\"\"\n";
    auto gvCode = GraphvizDFGGen(ir, ctx).print("kernel_preview");
    outputStream << gvCode << "\n";
    outputStream << "\"\"\"\n";
}

void PyTorchGen::generate(std::ostream& outputStream, std::string_view className, const AbstractAccess& forwardAccess, const PaddedConsts& consts) const {
    std::ostringstream code;

    auto printer = PythonCodePrinter { code, 0 };

    printer.writeLn("class {}(torch.nn.Module):", className);
    printer.indent([&] {
        printer.writeLn("def __init__(self, i):");
        printer.indent([&] {
            printer.writeLn("super().__init__()");
            printer.writeLn("self.id = i");
            printer.writeLn("self.shift_direction = (random.random() > 0.5) * 2 - 1");
            if (ir.inputTensors.size() == 1) {
                return;
            }
            printer.writeLn("self.weights = torch.nn.ParameterList([");
            printer.indent([&] {
                for (const Tensor& weight: ir.inputTensors | std::views::drop(1)) {
                    auto concretizedShape = concretize(weight.output(), consts.padded);
                    // If we want to adjust dimension order, we must defer this. TODO
                    printer.writeLn("torch.randn([{}]),", fmt::join(concretizedShape, ", "));
                }
            });
            printer.writeLn("])");
        });
    });
    printer.writeLn();
    printer.indent([&] {
        printer.writeLn("def forward(self, x):");
        printer.indent([&] {
            // We need to pad the input tensor.
            padInputTensor(printer, consts);
            // Load the weights.
            loadWeights(printer);
            for (const Tensor& tensor: topologicallyOrderedTensors) {
                SubgraphGen gen { ctx, graph, tensorNames, printer, tensor };
                gen.generate(consts.padded);
            }
            cropOutputTensor(printer, consts);
            applyDivision(printer, consts.padded, forwardAccess);
            printer.writeLn("return y");
        });
    });
    printer.writeLn();

    outputStream << code.str();
}

void PyTorchGen::generateSingle(const std::filesystem::path& outputPath, std::string_view className, const TensorView& tensorView, const std::map<std::string, std::size_t>& mappings) const {
    std::filesystem::create_directories(outputPath.parent_path());
    std::ofstream file { outputPath };
    PaddedConsts consts;
    consts.unpadded = ctx.realizeConsts(mappings);
    consts.padded = tensorView.computePadding(ctx, graph, consts.unpadded);
    generatePrelude(file);
    generate(file, className, tensorView.getForwardAccess(), consts);
}

} // namespace kas
