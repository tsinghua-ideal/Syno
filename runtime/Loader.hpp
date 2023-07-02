#pragma once

#include <filesystem>
#include <string>
#include <vector>


namespace at {

class Tensor;

} // namespace at

namespace kas {

// This is weird, we cannot include `KAS/CodeGen/Common.hpp` but have to paste it here.
// TODO: Include `KAS/CodeGen/Common.hpp`.
struct LoaderArgs {
    std::filesystem::path path;
    std::string symbol;
    bool cuda;
    std::size_t countInputs;
    std::size_t countKernels;
    std::vector<std::size_t> validPlaceholdersIndices;
};

class Loader {
    bool cuda;
    std::size_t countInputs;
    void *handle;
    std::vector<std::size_t> validPlaceholdersIndices;
    std::vector<void *> forwardPipelines;
    std::vector<void *> backwardPipelines;

    void call(const std::size_t expectedCountBuffers, void *pipeline, const std::vector<at::Tensor *>& buffers) const;

public:
    // Loads the dynamic library, and retrieves the function pointers.
    Loader(const LoaderArgs& args);
    Loader(const Loader&) = delete;
    Loader(Loader&&) = delete;
    // Calls the forward pipeline.
    void forward(std::size_t index, const std::vector<at::Tensor *>& buffers) const;
    // Calls the backward pipeline.
    void backward(std::size_t index, const std::vector<at::Tensor *>& buffers) const;
    // Unloads the dynamic library.
    ~Loader();
};

} // namespace kas
