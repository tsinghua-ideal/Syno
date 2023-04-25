#include <memory>

#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::unique_ptr<TensorView> FinalizeOp::buildTensorView() const {
    return std::make_unique<TensorView>(tensors);
}

std::string FinalizeOp::description(const BindingContext& ctx) const {
    return TensorArrayToString(tensors, ctx);
}

bool FinalizeOp::Prune(const std::vector<Graph::ConnectedComponent>& components, const std::vector<Interface>& trial) {
    std::map<Dimension, std::size_t, Dimension::AddressLessThan> dim2tensorId;
    for (std::size_t tId = 0; tId < trial.size(); ++tId) {
        for (auto&& dim: trial[tId]) {
            dim2tensorId[dim] = tId;
        }
    }

    // Early reduction analysis. For identity-mapped, sum-reduced, if a weight needs early reduction, then it is not canonical, which means we need to prune. TODO: if more types are added, change this.
    // We need to first identify the connected components in the indirected graph. If in a connected component, all output iterators are Sum, and all input iterators come from exactly one tensor, then this means we can do early reduction. For weight tensors, it is not reasonable to have early reduction, because this is pointless.
    for (auto&& component: components) {
        bool allSum = std::ranges::all_of(component.outputs, [](const Dimension& dim) {
            return dim.is(DimensionType::MapReduce);
        });
        if (!allSum) continue;
        const std::size_t tId = dim2tensorId.at(component.inputs.at(0));
        bool sameTensor = std::ranges::all_of(component.inputs | std::views::drop(1), [&](const Dimension& dim) {
            return tId == dim2tensorId.at(dim);
        });
        if (sameTensor) {
            // If this is from input tensor, then we can do early reduction to reduce FLOPs. TODO
            // But if this is from weight tensor, prune.
            if (tId != 0) {
                return true;
            }
        }
    }
    return false;
}

std::size_t FinalizeOp::CountSuccesses = 0;
std::size_t FinalizeOp::CountFailures = 0;
std::size_t FinalizeOp::CountLegalFinalizations = 0;
std::size_t FinalizeOp::CountConflictingColors = 0;
std::size_t FinalizeOp::CountPrunedFinalizations = 0;
std::vector<FinalizeOp> FinalizeOp::Generate(const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options) {
    Graph::Builder builder;
    builder.addTopmost(outputShape.items | std::views::transform(ColoredDimension::Projection{}));
    Graph graph = builder.build();
    auto components = graph.computeConnectedComponents();

    std::vector<FinalizeOp> result;
    const auto& desired = options.desired;
    auto recursion = [&](const auto& self, std::size_t nextIndex, std::vector<std::size_t> mappings) -> void {
        if (nextIndex == desired.size()) {
            std::vector<Interface> tensors { {} };
            auto& inputTensor = tensors.back();
            inputTensor.reserve(desired.size());
            auto used = std::vector<bool>(outputShape.size(), false);
            for (auto mapping: mappings) {
                inputTensor.emplace_back(outputShape[mapping]);
                used[mapping] = true;
            }
            if (options.maximumTensors == 1) {
                // Check that there is no excessive dimension.
                for (std::size_t i = 0; i < used.size(); ++i) {
                    if (!used[i]) {
                        return;
                    }
                }
            } else if (options.maximumTensors == 2) {
                if (outputShape.size() != desired.size()) {
                    // Add the dimensions to weight.
                    tensors.emplace_back();
                    auto& weightTensor = tensors.back();
                    for (std::size_t i = 0; i < used.size(); ++i) {
                        if (!used[i]) {
                            auto& cDim = outputShape.items[i];
                            if (!cDim.isUnknown() && cDim.color != Colors::Second) {
                                ++CountConflictingColors;
                                return; // Conflicting color!
                            }
                            weightTensor.emplace_back(outputShape[i]);
                        }
                    }
                }
            } else {
                KAS_UNIMPLEMENTED("maximumTensors > 2 not supported.");
            }
            if (!colors.isConsistent() || !Colors::CheckFinalization(tensors)) {
                ++CountConflictingColors;
                return;
            }
            if (Prune(components, tensors)) {
                ++CountPrunedFinalizations;
                return;
            }
            result.emplace_back(std::move(tensors));
            return;
        }
        const auto& desiredDim = desired[nextIndex];
        for (std::size_t i = 0; i < outputShape.size(); ++i) {
            auto& cDim = outputShape.items[i];
            if ((cDim.isUnknown() || cDim.color == Colors::First) && cDim.size() == desiredDim) {
                mappings.push_back(i);
                self(self, nextIndex + 1, mappings);
                mappings.pop_back();
            }
        }
    };
    recursion(recursion, 0, {});
    CountLegalFinalizations += result.size();
    if (result.empty()) {
        ++CountFailures;
    } else {
        ++CountSuccesses;
    }
    return result;
}

} // namespace kas
