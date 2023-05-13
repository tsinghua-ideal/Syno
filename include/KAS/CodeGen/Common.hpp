#pragma once

#include <filesystem>

#include "Halide.h"


namespace kas {

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

// Link the objects into a shared library.
int LinkObjects(const std::filesystem::path& dir, const std::string& soName, const std::vector<std::string>& objects);

} // namespace kas
