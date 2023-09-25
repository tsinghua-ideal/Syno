#include <fstream>
#include <ranges>

#include "KAS/CodeGen/GraphvizGen.hpp"
#include "KAS/CodeGen/PyTorchGen.hpp"
#include "KAS/Core/Dimension.hpp"
#include "KAS/Transforms/Transforms.hpp"
#include "KAS/Utils/Ranges.hpp"


namespace kas {

EinsumTensorContractor& EinsumTensorContractor::contract() {
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
    return *this;
}

Graph::DimensionSet PerformViewsIRPass::ViewPerformer::einsumContract() {
    return
        EinsumTensorContractor(
            subgraph.getGraph(),
            remainingShares
        )
        .with(subgraph.getTop().value())
        // Collect expansions.
        .with(
            subgraph.getOps()
            | std::views::filter([](const PrimitiveOp *op) {
                return op->getType() == DimensionType::Expand;
            })
            | std::views::transform([](const PrimitiveOp *op) -> const Dimension& {
                return dynamic_cast<const ExpandOp *>(op)->output;
            })
        )
        .contract()
        .dump();
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

PerformViewsIRPass::ViewPerformer::ViewPerformer(const Graph& graph, Tensor& tensor):
    DependentCutSetDiscoverer(graph, tensor.inputs().at(0).output()),
    tensor(tensor),
    subgraph(this->tensor.buildConstrainedGraph(graph))
{
    std::ranges::copy(
        subgraph.getOps()
        | std::views::filter([](const PrimitiveOp *op) {
            return op->getType() == DimensionType::Share;
        })
        | std::views::transform([](const PrimitiveOp *op) {
            return static_cast<const ShareOp *>(op);
        }),
        std::inserter(remainingShares, remainingShares.begin())
    );
}

void PerformViewsIRPass::ViewPerformer::apply() {
    // First we find out the einsum-contracted interface.
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
    // But we only need to find inputs of ShareOp's. They are all that are necessary.
    struct EinsumCutSetDiscoverer: public DependentCutSetDiscoverer {
        using DependentCutSetDiscoverer::DependentCutSetDiscoverer;
        void excludeHook(const PrimitiveOp *op) override {
            KAS_ASSERT(op->getType() != DimensionType::Share);
        }
    };
    auto discoverer = EinsumCutSetDiscoverer(graph, einsumContractionResult);
    for (const auto& [dim, state]: visited) {
        if (state == State::Collected && dim.is(DimensionType::Share)) {
            discoverer.include(dim);
        }
    }

    // With the newly collected cut set as the stage tensor, build a view.
    // TODO: make this a helper function.
    auto viewTensor = TensorImpl::CreateView(
        tensor.inputs(),
        discoverer.build(),
        std::vector<const Reduce *>{}
    );
    // Replace the input tensor.
    tensor.getInputs() = { viewTensor };

    // Recursively perform another view.
    ViewPerformer(graph, tensor).shouldWarn(warn).apply();
}

PerformViewsIRPass::PerformViewsIRPass(IR& ir): graph(ir.buildGraph()), ir(ir) {}

void PerformViewsIRPass::apply() {
    ir.forEach([&](Tensor& tensor) {
        if (tensor.hasContraction()) {
            ViewPerformer(graph, tensor).apply();
        }
    });
}

std::pair<Dimension, std::size_t> PyTorchGen::SubgraphGen::assignShare(Dimension dim) {
    auto subscript = newSubscript();
    auto discoverer = ShareBlockDiscoverer(graph, dim, bottommost, [&](const Dimension& d) {
        assignSubscript(d, subscript);
    });
    return { discoverer.traverse(), subscript };
}

std::vector<Dimension> PyTorchGen::SubgraphGen::performContraction(std::string_view name) {
    KAS_ASSERT(!tensor.isInputTensor());

    auto isToBeShared = [&](const Dimension& dim) {
        return
            // This is a ShareOp.
            dim.type() == DimensionType::Share &&
            // But we have to check whether it is preserved in the output.
            !bottommost.contains(dim);
    };

    // Consider the special case where we do not need any einsum.
    if (
        const auto& inputTensor = tensor.inputs()[0];
        tensor.inputs().size() == 1 && // We only have one input tensor so we do not need to do contraction or outer product,
        std::ranges::none_of(inputTensor.output(), [&](const Dimension& dim) {
            return
                // Since we have added reduction in the end of each subgraph, and there is no possibility that we have a remaining reduction, we do not need to check for this condition.
                // dim.type() == DimensionType::Reduce || // there is no reduction.
                isToBeShared(dim); // and there is no Share to be performed.
        })
    ) {
        printer.writeLn("{} = {}", name, tensorNames.at(inputTensor));
        printer.writeLn();
        return inputTensor.output();
    }

    // First we only perform share. Reductions are later considered.
    std::vector<std::pair<Dimension, std::size_t>> realInput;
    std::vector<std::vector<std::size_t>> inputsSubscripts;
    for (const Tensor& inputTensor: tensor.inputs()) {
        inputsSubscripts.emplace_back();
        auto& inputSubscripts = inputsSubscripts.back();
        for (const Dimension& dim: inputTensor.output()) {
            if (auto it = subscripts.find(dim); it != subscripts.end()) {
                // This is a shared dimension, and we are sure that this is already in realInput.
                inputSubscripts.emplace_back(it->second);
                continue;
            }
            if (isToBeShared(dim)) {
                // Assign with new subscript.
                auto [realDim, subscript] = assignShare(dim);
                inputSubscripts.emplace_back(subscript);
                realInput.emplace_back(realDim, subscript);
            } else {
                // We need to pass it down.
                auto subscript = newSubscript();
                inputSubscripts.emplace_back(subscript);
                realInput.emplace_back(dim, subscript);
            }
        }
    }

    // Perform reduction. By removing all required Reduce in realInput.
    // This only works for products. TODO: make this work for addition.
    std::vector<std::pair<Dimension, std::size_t>> newRealInput;
    for (auto&& pair: realInput) {
        auto reduction = pair.first.tryAs<Reduce>();
        if (reduction && remainingReductions.contains(reduction)) {
            // This is a reduction.
            remainingReductions.erase(reduction);
        } else {
            newRealInput.emplace_back(std::move(pair));
        }
    }
    realInput = std::move(newRealInput);

    // Then contract.
    printer.writeLn(
        R"code({} = torch.einsum("{} -> {}", {}))code",
        name,
        fmt::join(inputsSubscripts | std::views::transform(&ToEinsteinNotation<const std::vector<std::size_t>&>), ", "),
        ToEinsteinNotation(realInput | std::views::transform(&std::pair<Dimension, std::size_t>::second)),
        fmt::join(tensor.inputs() | std::views::transform([&](const Tensor& inputTensor) -> decltype(auto) {
            return tensorNames.at(inputTensor);
        }), ", ")
    );
    printer.writeLn();

    return ranges::to<std::vector<Dimension>>(realInput | std::views::transform(&std::pair<Dimension, std::size_t>::first));
}

PyTorchGen::SubgraphGen::OpLower::OpLower(const BindingContext& ctx, const Graph& graph, PythonCodePrinter& printer, const ConcreteConsts& consts, std::vector<Dimension>& interface, const Tensor& tensor, std::set<const Reduce *>& remainingReductions, const std::string& name):
    ctx { ctx }, graph { graph }, printer { printer }, consts { consts }, interface { interface }, tensor { tensor }, remainingReductions(remainingReductions), name { name }
{
    for (std::size_t index = 0; const auto& outputDim: this->tensor.output()) {
        outputSet.emplace(outputDim, index);
        // It is possible that there are Reduce's in output.
        // KAS_ASSERT(outputDim.type() != DimensionType::Reduce);
        ++index;
    }
}

void PyTorchGen::SubgraphGen::OpLower::lower() {
    while (true) {
        bool changed = false;
        for (Dimension dim: interface) {
            // Lower an Op.
            if (!outputSet.contains(dim)) {
                // Try lowering.
                if (graph.visitAlong(dim, Direction::Down).match(*this)) {
                    // Only upon successful visit, break and start another iteration.
                    changed = true;
                    break;
                }
            }
        }
        if (!changed) break; // All Op's lowered.
    }

    // Now reduce the remaining dimensions.
    if (!remainingReductions.empty()) {
        printer.writeLn("# Reduce the remaining dimensions.");
        std::vector<std::size_t> reducedIndices;
        std::vector<Dimension> newInterface;
        for (std::size_t i = 0; i < interface.size(); ++i) {
            if (auto reduction = interface[i].tryAs<Reduce>(); reduction) {
                if (auto it = remainingReductions.find(reduction); it != remainingReductions.end()) {
                    // This is a dimension to be reduced.
                    reducedIndices.emplace_back(i);
                    remainingReductions.erase(it);
                    continue;
                }
            }
            // This is a dimension to be kept.
            newInterface.emplace_back(interface[i]);
        }
        printer.writeLn("{0} = torch.sum({0}, dim=({1}, ))", name, fmt::join(reducedIndices, ", "));
        interface = std::move(newInterface);
    }

    // Finished.
    KAS_ASSERT(remainingReductions.empty());
    KAS_ASSERT(interface.size() == tensor.output().size());

    // Well, we need to permute the interface so that it matches the output of this subgraph.
    // Maybe we can alter the output to get better performance? TODO
    std::vector<std::size_t> indices(interface.size(), std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 0; const Dimension& dim: interface) {
        indices[outputSet.at(dim)] = i;
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
    }
}

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
        channelSize = std::transform_reduce(interface.begin() + 1, interface.begin() + heightIndexInInterface, static_cast<std::size_t>(1), std::multiplies<std::size_t>(), [this](const Dimension& dim) {
            return concretize(dim.size());
        });
    }
    const std::size_t widthSize = std::transform_reduce(interface.begin() + heightIndexInInterface + 1, interface.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>(), [this](const Dimension& dim) {
        return concretize(dim.size());
    });
    printer.writeLn("{0} = torch.reshape({0}, ({1}, {2}, {3}, {4}, ))", name, batchSize, channelSize, heightSize, widthSize);

    return { batchSize, channelSize, heightSize, widthSize };
}

void PyTorchGen::SubgraphGen::OpLower::visit(const MergeOp& op) {
    const auto lhs = op.getInputL(), rhs = op.getInputR();
    std::size_t lhsIndex = std::distance(interface.begin(), std::ranges::find(interface, lhs));
    std::size_t rhsIndex = std::distance(interface.begin(), std::ranges::find(interface, rhs));
    const std::size_t length = interface.size();
    if (lhsIndex == length || rhsIndex == length) {
        // We need to defer the lowering of this.
        return;
    }
    successfulVisit = true;
    KAS_ASSERT(lhsIndex != rhsIndex);

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
void PyTorchGen::SubgraphGen::OpLower::visit(const ShareOp& op) {
    // No progress. successfulVisit == false.
}
void PyTorchGen::SubgraphGen::OpLower::visit(const ShiftOp& op) {
    const auto [input, inputIndex] = getSingleInput(op);
    printer.writeLn("{0} = torch.roll({0}, {1}, {2})", name, -op.getShift(), inputIndex);
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

PyTorchGen::SubgraphGen::SubgraphGen(const BindingContext& ctx, const Graph& graph, const std::map<Tensor, std::string>& tensorNames, PythonCodePrinter& printer, const Tensor& tensor):
    ctx { ctx }, graph { graph }, tensorNames(tensorNames), printer { printer }, tensor { tensor }
{
    bottommost.insert(tensor.output().begin(), tensor.output().end());
    remainingReductions.insert(tensor.reductions().begin(), tensor.reductions().end());
}

void PyTorchGen::SubgraphGen::generate(const ConcreteConsts& consts) {
    const auto& name = tensorNames.at(tensor);

    // First perform contraction.
    auto interface = performContraction(name);

    // Next, perform expansions.
    auto subgraph = tensor.buildConstrainedGraph(graph);
    const auto expansions = ranges::to<Graph::DimensionSet>(
        subgraph.getOps()
        | std::views::transform([](const PrimitiveOp *op) { return dynamic_cast<const ExpandOp *>(op); })
        | std::views::filter([](const ExpandOp *expand) { return expand != nullptr; })
        | std::views::transform(Expand::PointerToDimension{})
    );
    if (!expansions.empty()) {
        // TODO: optimize locality.
        auto currentShape = ShapeView(interface).eval<std::size_t>(consts);
        // We need to perform unsqueeze and expand to obtain the interface for the subgraph.
        printer.writeLn("# We need to unsqueeze and expand the input tensor.");
        for (std::size_t i = 0; i < expansions.size(); ++i) {
            currentShape.emplace_back(1);
        }
        printer.writeLn("{0} = torch.reshape({0}, ({1}, ))", name, fmt::join(currentShape, ", "));
        for (std::size_t i = interface.size(); const Dimension& dim: expansions) {
            interface.emplace_back(dim);
            currentShape[i] = dim.size().eval<std::size_t>(consts);
            ++i;
        }
        printer.writeLn("{0} = {0}.expand({1})", name, fmt::join(currentShape, ", "));
        printer.writeLn();
    }

    // Now we have got the input. Time to work on it. The objective is to reach `tensor.output()`.
    auto opLower = OpLower { ctx, graph, printer, consts, interface, tensor, remainingReductions, name };
    opLower.lower();
}

std::vector<std::size_t> PyTorchGen::concretize(const std::vector<Dimension>& interface, const ConcreteConsts& consts) const {
    return ShapeView(interface).eval<std::size_t>(consts);
}

PyTorchGen::PyTorchGen(const BindingContext& ctx, const IR& ir):
    ctx { ctx },
    ir { ir.copy() },
    graph { this->ir.buildGraph() }
{
    PerformViewsIRPass(this->ir).apply();
    for (std::size_t index = 0; const auto& tensor: this->ir.inputTensors) {
        declare(tensor, "in_" + std::to_string(index));
        ++index;
    }
    auto dfs = [&](const auto& self, const Tensor& tensor) -> void {
        if (declared(tensor)) {
            return;
        }
        declare(tensor);
        for (const Tensor& source: tensor.inputs()) {
            self(self, source);
        }
        if (!tensor.isInputTensor()) {
            topologicallyOrderedTensors.emplace_back(tensor);
        }
    };
    dfs(dfs, this->ir.outputTensor);
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
        printer.writeLn("def __init__(self):");
        printer.indent([&] {
            printer.writeLn("super().__init__()");
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
    consts.padded = tensorView.computePadding(ctx, consts.unpadded);
    generatePrelude(file);
    generate(file, className, tensorView.getForwardAccess(), consts);
}

} // namespace kas
