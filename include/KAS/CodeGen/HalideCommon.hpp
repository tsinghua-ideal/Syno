#pragma once

#include "Halide.h"

#include "KAS/Utils/Common.hpp"


namespace kas {

inline Halide::Target GetHostTarget(bool useGPU, bool withRuntime) {
    auto t = Halide::get_host_target();
    // t.with_feature(Halide::Target::Debug);
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
    KAS_ASSERT(Halide::host_supports_target_device(t));
    return t;
}

} // namespace kas
