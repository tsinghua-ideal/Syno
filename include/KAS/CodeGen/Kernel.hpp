#pragma once

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "KAS/CodeGen/Common.hpp"
#ifdef KAS_USE_HALIDE
#include "KAS/CodeGen/HalideGen.hpp"
#endif
#include "KAS/CodeGen/Options.hpp"
#include "KAS/Core/TensorView.hpp"


namespace kas {

// In a directory:
//   - metadata.json
//   - nested_loops.h
//   - halide_schedule.h, if Halide is enabled
//   - kernels.so, if Halide is enabled
//   - kernel_graph.dot
//   - kernels.py
//   - kernels_tvm.py
//
// The functions in kernels.so are named as name_0, name_0_grad, name_1, name_1_grad, ...
// Other undocumented files are also generated.
struct KernelMetadata {
    struct PlaceholderMetadata {
        PaddedConsts consts;
        std::string constsDescription;
        std::vector<std::vector<std::size_t>> unpaddedInputsShapes;
        std::vector<std::vector<std::size_t>> paddedInputsShapes;
        std::vector<std::size_t> unpaddedOutputShape;
        std::vector<std::size_t> paddedOutputShape;
        std::size_t flops;
    };
    std::string name;
    std::vector<std::string> inputsShapes;
    std::string outputShape;
    bool halide;
    bool cuda;
    std::size_t countInputs;
    std::vector<std::size_t> validPlaceholdersIndices;
    std::vector<PlaceholderMetadata> validPlaceholders;
    std::size_t countPlaceholders() const;
    std::size_t countKernels() const;
    const PlaceholderMetadata& getPlaceholder(std::size_t index) const;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(KernelMetadata::PlaceholderMetadata, consts, constsDescription, unpaddedInputsShapes, paddedInputsShapes, unpaddedOutputShape, paddedOutputShape, flops);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(KernelMetadata, name, inputsShapes, outputShape, halide, cuda, countInputs, validPlaceholdersIndices, validPlaceholders);

// This is for convenience. As a python interface, we need easy access to related methods of TensorView.
class Kernel {
protected:
    std::filesystem::path dir;
    KernelMetadata metadata;
    std::string nestedLoops;

    void loadMetadataAndNestedLoops();

public:
    // Generate and save to directory.
    Kernel(const BindingContext& ctx, const TensorView& tensorView, const std::vector<std::map<std::string, std::size_t>>& allMappings, CodeGenOptions options, const std::filesystem::path& dir, const std::string& name);

    // Load from file.
    Kernel(const std::filesystem::path& dir);

    const std::string& getName() const;
    const std::filesystem::path& getDirectory() const;

    const std::string& getNestedLoops() const;

    bool halide() const;
    bool cuda() const;
    std::size_t countPlaceholders() const;
    std::size_t countKernels() const;
    std::size_t getValidPlaceholderIndex(std::size_t index) const;

    const std::string& getConsts(std::size_t index) const;

    std::size_t getFLOPs(std::size_t index) const;
    std::size_t getTotalFLOPs() const;

    std::size_t getCountInputs() const;
    const std::vector<std::vector<std::size_t>>& getInputsShapes(bool padded, std::size_t index) const;
    const std::vector<std::size_t>& getOutputShape(bool padded, std::size_t index) const;

    LoaderParameters getLoaderArgs() const;
};

} // namespace kas
