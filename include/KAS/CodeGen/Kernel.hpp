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
    BindingContext& ctx;
    HalideGen gen;
    std::vector<PaddedConsts> paddedConsts;

public:
    Kernel(const TensorView& tensorView, BindingContext& ctx, const std::vector<std::map<std::string, std::size_t>>& allMappings, HalideGen::Options options);

    std::string toNestedLoops() const;

    // Generate headers and static libraries named as name_i.pytorch.h, and name_i.a in the specified directory.
    void generateOperator(const std::string& path, const std::string& name);
    void generateGraphviz(const std::string& path, const std::string& name);

    std::string getConsts(std::size_t index) const;

    std::vector<std::vector<std::size_t>> getInputsShapes(bool padded, std::size_t index) const;

    std::vector<std::size_t> getOutputShape(bool padded, std::size_t index) const;
};

} // namespace kas
