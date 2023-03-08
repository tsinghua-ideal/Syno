#include <memory>

#include "KAS/Core/Tensor.hpp"
#include "KAS/Search/Finalize.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

std::unique_ptr<TensorView> FinalizeOp::buildTensorView() const {
    return std::make_unique<TensorView>(tensors);
}

std::vector<FinalizeOp> FinalizeOp::Generate(const ColoredInterface& outputShape, const Colors& colors, GenerateOptions options) {
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
                                return; // Conflicting color!
                            }
                            weightTensor.emplace_back(outputShape[i]);
                        }
                    }
                }
            } else {
                KAS_UNIMPLEMENTED("maximumTensors > 2 not supported.");
            }
            if (!colors.checkFinalization(tensors)) {
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
    return result;
}

} // namespace kas
