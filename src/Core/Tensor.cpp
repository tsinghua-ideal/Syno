#include "KAS/Core/Tensor.hpp"
#include "KAS/Utils/Ranges.hpp"

namespace kas {

std::pair<std::vector<Tensor>, std::vector<Dimension>> Tensor::Builder::findTensorsWhichWeNeedToContract(const Tensor& tensor) const {
    // Get the initial interface.
    std::vector<Dimension> interface;
    std::vector<Tensor> contractedTensors { tensor };
    std::set<Dimension, Dimension::AddressLessThan> contractedDimensions;
    for (const Dimension& dim: tensor.output()) {
        // They must be ours.
        KAS_ASSERT(owner.at(dim) == tensor);
        if (dim.type() == DimensionType::Share) { // We need to find the Share block.
            if (!contractedDimensions.contains(dim)) {// We have not done this.
                bool rollback = false;
                const std::size_t previouslyContractedTensors = contractedTensors.size();
                auto discoverer = ShareBlockDiscoverer(graph, dim, [&](const Dimension& d) {
                    auto [_, inserted] = contractedDimensions.emplace(d);
                    KAS_ASSERT(inserted, "Cannot contract a dimension twice!");
                    if (auto it = owner.find(d); it != owner.end()) {
                        // Maybe we have find another tensor to contract!
                        if (std::ranges::find(contractedTensors, it->second) == contractedTensors.end()) {
                            // Collect the tensors to contract.
                            contractedTensors.emplace_back(it->second);
                        }
                    } else {
                        // It seems that there are still some unreached dimensions!
                        // Roll back! This cannot be contracted!
                        rollback = true;
                    }
                });
                if (!rollback) {
                    // Substitute the dimension with the contracted one.
                    interface.emplace_back(discoverer.traverse());
                } else {
                    KAS_DEBUG("Rolling back...");
                    // Undo all the changes.
                    ShareBlockDiscoverer(graph, dim, [&](const Dimension& d) {
                        contractedDimensions.erase(d);
                    }).traverse();
                    contractedTensors.resize(previouslyContractedTensors);
                    interface.emplace_back(dim);
                }
            }
        } else {
            interface.emplace_back(dim);
        }
    }
    // A special case: the output tensor must still be seen as Share.
    if (std::ranges::all_of(tensor.output(), [](const Dimension& dim) {
        return dim.type() == DimensionType::Iterator;
    })) {
        for (Dimension iterator: graph.getOutputIterators()) {
            if (auto it = owner.find(iterator); it != owner.end()) {
                // Maybe we have find another tensor to contract!
                if (std::ranges::find(contractedTensors, it->second) == contractedTensors.end()) {
                    // Collect the tensors to contract.
                    contractedTensors.emplace_back(it->second);
                }
            }
        }
    }
    // Now we have put the dimensions in input tensor in the interface.
    // Time to collect the dimensions in contracted tensors.
    for (const Tensor& other: contractedTensors | std::views::drop(1)) {
        for (const Dimension& dim: other.output()) {
            if (contractedDimensions.contains(dim)) {
                continue;
            }
            // Usually this will not happen.
            interface.emplace_back(dim);
        }
    }
    return { std::move(contractedTensors), std::move(interface) };
}

Subgraphs Tensor::Builder::build(const std::vector<std::vector<Dimension>>& rawInputTensors) {
    auto inputTensors = ranges::to<std::vector<Tensor>>(rawInputTensors
        | std::views::transform([this](const std::vector<Dimension>& tensor) {
            return TensorImpl::CreateInputTensor(*this, tensor);
        })
    );

    // Perform early reduction.
    std::vector<Tensor> workingTensors;
    for (std::size_t i = 0; i < rawInputTensors.size(); ++i) {
        workingTensors.emplace_back(TensorImpl::CreateTensorView(*this, { inputTensors[i] }, rawInputTensors[i]));
    }

    std::size_t counter = 0;
    // Then, find contractions, as long as
    while (
        workingTensors.size() > 1 // either there are tensors remaining to be contracted,
        || std::ranges::any_of(workingTensors.at(0).output(), [](const Dimension& dim) {
            return dim.type() != DimensionType::Iterator; // or we still need to perform some view or taking the diagonal of a tensor.
        })
    ) {
        auto [contractions, contractedInterface] = findTensorsWhichWeNeedToContract(workingTensors.at(0));
        Tensor contractedTensor = TensorImpl::CreateTensorView(*this, contractions, std::move(contractedInterface));
        auto [removeBegin, removeEnd] = std::ranges::remove_if(workingTensors, [&contractions](const Tensor& tensor) {
            return std::ranges::find(contractions, tensor) != contractions.end();
        });
        workingTensors.erase(removeBegin, removeEnd);
        workingTensors.insert(workingTensors.begin(), std::move(contractedTensor));
        ++counter;
        KAS_ASSERT(counter < 10, "Too many contractions!");
    }

    // Great! We are done.
    KAS_ASSERT(workingTensors.size() == 1);
    auto& outputTensor = workingTensors[0];
    std::vector<Dimension> expectedOutput;
    std::ranges::copy(graph.getOutputIterators(), std::back_inserter(expectedOutput));
    // Do not forget to permute the dimensions.
    outputTensor.inner->adjustOutputOrder(expectedOutput);
    return { std::move(inputTensors), std::move(outputTensor) };
}

const std::vector<Tensor>& Tensor::inputs() const {
    return inner->inputs;
}
const std::vector<Dimension>& Tensor::output() const {
    return inner->output;
}
bool Tensor::isInputTensor() const {
    return inner->isInputTensor();
}

std::string Tensor::toString(const BindingContext& ctx) const {
    auto outputString = ShapeView(output()).toString(ctx);
    if (isInputTensor()) {
        return outputString;
    }
    return fmt::format("({} -> {})", fmt::join(inputs() | std::views::transform([&](const Tensor& t) { return t.toString(ctx); }), ", "), outputString);
}

std::string Tensor::debugToString() const {
    return BindingContext::ApplyDebugPublicCtx(&Tensor::toString, *this);
}

Tensor TensorImpl::CreateInputTensor(Tensor::Builder& builder, const std::vector<Dimension>& dimensions) {
    Tensor result = std::shared_ptr<TensorImpl>(new TensorImpl(dimensions));
    for (const auto& dim: dimensions) {
        auto [it, inserted] = builder.owner.emplace(dim, result);
        KAS_ASSERT(inserted);
    }
    return result;
}

Tensor TensorImpl::CreateTensorView(Tensor::Builder& builder, const std::vector<Tensor>& inputs, std::vector<Dimension> interface) {
    // We cannot determine the output yet.
    Tensor result = std::shared_ptr<TensorImpl>(new TensorImpl(inputs, std::vector<Dimension>{}));

    bool anyProgressAtAll = inputs.size() > 1;
    // Although we have contracted the tensors, we actually need to do the reductions.
    std::size_t beforeReduction = interface.size();
    auto [removeBegin, removeEnd] = std::ranges::remove_if(interface, [](const Dimension& dim) {
        return dim.type() == DimensionType::MapReduce;
    });
    interface.erase(removeBegin, removeEnd);
    anyProgressAtAll |= beforeReduction > interface.size();

    // Now, we want to traverse the graph and expand the interface as far as possible.
    // Stop only at MergeLikeOp::Input. Mark it as ours, and continue. If both branches of MergeLikeOp has been marked as ours, we can proceed. Otherwise if the two branches have different owners, this must be a ShareOp, in which case we need to contract. But that is left to the next tensor view.

    struct Visitor final: public DimVisitor {
        Tensor::Builder& builder;
        const Tensor& newTensor;
        std::vector<Dimension>& interface;
        bool hasProgress = false;
        Visitor(Tensor::Builder& builder, const Tensor& newTensor, std::vector<Dimension>& interface): builder { builder }, newTensor { newTensor }, interface { interface } {}
        using DimVisitor::visit;
        void visit(const RepeatLikeOp::Input& dim) {
            auto input = dim.getOp()->getInput();
            auto it = std::ranges::find(interface, input);
            auto output = dim.getOp()->output;
            *it = output;
            visit(output, false);
        }
        void visit(const SplitLikeOp::Input& dim) {
            auto input = dim.getOp()->getInput();
            auto it = std::ranges::find(interface, input);
            auto index = std::distance(interface.begin(), it);
            auto outputRhs = dim.getOp()->outputRhs, outputLhs = dim.getOp()->outputLhs;
            *it = outputRhs;
            interface.insert(it, outputLhs);
            if (dim.type() == DimensionType::Unfold) {
                // This is a workaround for PyTorch codegen.
                // TODO: Handle this in PyTorch codegen.
                std::swap(interface[index], interface[index + 1]);
            }
            visit(outputLhs, false);
            visit(outputRhs, false);
        }
        void visit(const MergeLikeOp::Input& dim) {
            if (dim.type() == DimensionType::Share) {
                // Stop here.
                return;
            }
            auto other = dim.getOther();
            if (builder.owner.contains(other)) {
                // We can proceed.
                auto indexThis = std::distance(interface.begin(), std::ranges::find(interface, &dim));
                auto indexOther = std::distance(interface.begin(), std::ranges::find(interface, other));
                auto output = dim.getOp()->output;
                interface[std::max(indexThis, indexOther)] = output;
                interface.erase(interface.begin() + std::min(indexThis, indexOther));
                visit(output, false);
            }
        }
        void visit(Dimension dim, bool isSource) {
            if (!isSource) {
                if (builder.owner.contains(dim)) {
                    // Visited.
                    return;
                }
                hasProgress = true;
            }
            // Mark as ours.
            builder.owner.insert_or_assign(dim, newTensor);
            // Match.
            DimVisitor::visit(dim);
        }
    };
    auto visitor = Visitor { builder, result, interface };
    do {
        visitor.hasProgress = false;
        for (const auto& dim: interface) {
            visitor.visit(dim, true);
            if (visitor.hasProgress) {
                anyProgressAtAll = true;
                break;
            }
        }
    } while (visitor.hasProgress);

    if (!anyProgressAtAll) {
        // If this does not cause any change, just return the original input.
        KAS_ASSERT(inputs.size() == 1);
        for (auto&& [dim, t]: builder.owner) {
            if (t == result) {
                t = inputs[0];
            }
        }
        return inputs[0];
    }

    result.inner->output = std::move(interface);
    return result;
}

void TensorImpl::adjustOutputOrder(const std::vector<Dimension>& expectedOutput) {
    KAS_ASSERT(output.size() == expectedOutput.size());
    std::set<Dimension, Dimension::AddressLessThan> expectedOutputSet(expectedOutput.begin(), expectedOutput.end());
    KAS_ASSERT(std::ranges::all_of(output, [&](const Dimension& dim) {
        return expectedOutputSet.contains(dim);
    }));
    output = expectedOutput;
}

} // namespace kas
