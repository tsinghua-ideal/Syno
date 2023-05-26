#include <memory>

#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Search/Sample.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::unique_ptr<TensorView> FinalizeOp::buildTensorView(const std::vector<FixedDimension>& fixed) const {
    if (fixed.empty()) {
        return std::make_unique<TensorView>(tensors);
    }
    std::vector<Interface> tensors;
    std::ranges::copy(this->tensors, std::back_inserter(tensors));
    auto& inputTensor = tensors.at(0);
    for (const auto& [index, dim]: fixed) {
        // Given the fact that fixed is sorted.
        inputTensor.insert(inputTensor.begin() + index, dim);
    }
    return std::make_unique<TensorView>(tensors);
}

std::string FinalizeOp::description(const BindingContext& ctx) const {
    return TensorArrayToString(tensors, ctx);
}

bool FinalizeOp::Prune(const std::vector<Graph::ConnectedComponent>& components, const std::vector<Interface>& trial) {
    std::map<Dimension, std::size_t, Dimension::AddressLessThan> dim2tensorId;
    for (std::size_t tId = 0; tId < trial.size(); ++tId) {
        for (auto&& dim: trial[tId]) {
            if (tId >= 1) {
                // We also check for uncanonical cases here.
                // In a single tensor, there must not be both inputs of MergeOp, or SplitOp, or ShiftOp.
                auto dimType = dim.type();
                switch (dimType) {
                case DimensionType::Merge: {
                    const auto& merge = dim.as<MergeOp::Input>();
                    if (auto it = dim2tensorId.find(merge.getOther()); it != dim2tensorId.end()) {
                        if (it->second == tId) {
                            return true;
                        }
                    }
                    break;
                }
                case DimensionType::Split:
                case DimensionType::Shift:
                    return true;
                default:
                    break;
                }
            }
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

namespace {

struct CollectedTensorFragments {
    std::vector<std::size_t> mappings;
    Color color;
    bool canAccept(std::size_t index, const Color& color) const {
        // Collect tags.
        return std::ranges::find(mappings, index) == mappings.end() && this->color.disjoint(color);
    }
    void accept(std::size_t index, const Color& color) {
        mappings.emplace_back(index);
        // Merge tags.
        this->color.merge(color);
    }
    std::vector<Dimension> toTensor(const ColoredInterface& interface, auto&& callback) const {
        std::vector<Dimension> result;
        result.reserve(mappings.size());
        for (auto mapping: mappings) {
            result.emplace_back(interface[mapping].dimension);
            callback(mapping);
        }
        return result;
    }
};

} // namespace

std::vector<FinalizeOp> FinalizeOp::Generate(const ColoredInterface& interface, const Graph& graph, const GenerateOptions& options) {
    ++CountGenerateInvocations;

    // First we perform a basic check. If any Dimension is data-discarding, then it is not a legal kernel.
    if (std::ranges::any_of(interface, [](const ColoredDimension& cdim) { return cdim.color.isDataDiscarding(); })) {
        ++CountFailedInvocations;
        return {};
    }

    // Compute connected components for early reduction analysis.
    auto components = graph.computeConnectedComponents();

    std::vector<FinalizeOp> result;
    const auto& desired = options.desired;

    auto buildBesideInputTensor = [&](const CollectedTensorFragments& inputCandidate) {
        std::vector<Interface> tensors { {} };
        auto& inputTensor = tensors.back();
        auto used = std::vector<bool>(interface.size(), false);
        inputTensor = inputCandidate.toTensor(interface, [&](std::size_t index) {
            used[index] = true;
        });
        if (options.maximumTensors == 1) {
            // Check that there is no excessive dimension.
            for (std::size_t i = 0; i < used.size(); ++i) {
                if (!used[i]) {
                    return;
                }
            }
        } else if (options.maximumTensors == 2) {
            if (interface.size() != desired.size()) {
                // Add the dimensions to weight.
                tensors.emplace_back();
                Color weightColors;
                auto& weightTensor = tensors.back();
                for (std::size_t i = 0; i < used.size(); ++i) {
                    if (!used[i]) {
                        auto& [dim, color] = interface[i];
                        if (!weightColors.disjoint(color)) {
                            // Conflicting color!
                            ++CountConflictingColors;
                            return;
                        }
                        weightColors.merge(color);
                        weightTensor.emplace_back(interface[i].dimension);
                    }
                }
            }
        } else {
            KAS_UNIMPLEMENTED("maximumTensors > 2 not supported.");
        }
        if (!Color::CheckFinalization(tensors)) {
            // We really should avoid this!
            KAS_WARNING("Finalization with conflicting colors generated!");
            ++CountConflictingColors;
            return;
        }
        if (Prune(components, tensors)) {
            ++CountPrunedFinalizations;
            return;
        }
        result.emplace_back(std::move(tensors));
    };

    auto collectInputDimensions = [&](const auto& self, std::size_t nextIndex, const CollectedTensorFragments& fragments) -> void {
        if (nextIndex == desired.size()) {
            // We have collected the full input shape. Now build the weights.
            buildBesideInputTensor(fragments);
            return;
        }
        const auto& desiredDimSize = desired[nextIndex];
        for (std::size_t i = 0; i < interface.size(); ++i) {
            auto&& [dim, color] = interface[i];
            if (color.countRightTags() > 0) {
                // For canonicalization, we can assume that the input tensor only has left tags.
                continue;
            }
            if (dim.size() == desiredDimSize && fragments.canAccept(i, color)) {
                auto newFragments = fragments;
                newFragments.accept(i, color);
                self(self, nextIndex + 1, newFragments);
            }
        }
    };
    collectInputDimensions(collectInputDimensions, 0, {});

    CountLegalFinalizations += result.size();
    if (result.empty()) {
        ++CountFailedInvocations;
    } else {
        ++CountSuccessfulInvocations;
    }
    return result;
}

} // namespace kas
