#pragma once

#include <filesystem>
#include <vector>

#ifdef KAS_USE_HALIDE
#include "Halide.h"
#endif


namespace kas {

#ifdef KAS_USE_HALIDE
inline Halide::Target GetHostTarget(bool useGPU, bool withRuntime) {
    auto t = Halide::get_host_target();
    if (useGPU) {
        t = t
            .with_feature(Halide::Target::CUDA)
            .with_feature(Halide::Target::UserContext); // UserContext is required by CUDA streams.
    }
    if (!withRuntime) {
        // If not JIT, we will use Loader (which is linked against a single Halide runtime) to load the generated kernel.
        t = t
            .with_feature(Halide::Target::NoRuntime);
    }
    return t;
}
#endif

// Link the objects into a shared library.
int LinkObjects(const std::filesystem::path& dir, const std::string& soName, const std::vector<std::string>& objects);

struct LoaderParameters {
    std::filesystem::path path;
    std::string symbol;
    bool cuda;
    std::size_t countInputs;
    std::size_t countKernels;
    std::vector<std::size_t> validPlaceholdersIndices;
};

} // namespace kas
