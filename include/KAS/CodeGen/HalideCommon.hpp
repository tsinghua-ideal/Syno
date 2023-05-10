#pragma once

#include "Halide.h"

#include "KAS/Utils/Common.hpp"


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

} // namespace kas
