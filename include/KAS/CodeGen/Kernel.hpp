#pragma once

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include "KAS/CodeGen/HalideGen.hpp"
#include "KAS/Core/Tensor.hpp"


namespace kas {

// This is for convenience. As a python interface, we need easy access to related methods of TensorView.
class Kernel {
protected:
    const TensorView& tensorView;
    const BindingContext& ctx;
    HalideGen gen;
    std::vector<PaddedConsts> paddedConsts;

public:
    Kernel(const TensorView& tensorView, const BindingContext& ctx, const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options);
    const TensorView& getTensorView() const { return tensorView; }

    std::string toNestedLoops() const;

    // Generate shared library named as name.so in the specified directory. The functions are named as name_0, name_0_grad, name_1, name_1_grad, ...
    void generateOperator(const std::string& dir, const std::string& name);
    void generateGraphviz(const std::string& dir, const std::string& name);

    std::string getConsts(std::size_t index) const;

    std::size_t getFLOPs(std::size_t index) const;
    std::size_t getTotalFLOPs() const;

    std::size_t getCountInputs() const;
    std::vector<std::vector<std::size_t>> getInputsShapes(bool padded, std::size_t index) const;
    std::vector<std::size_t> getOutputShape(bool padded, std::size_t index) const;
};

} // namespace kas
