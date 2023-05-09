#include <dlfcn.h>
#include <fmt/core.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <HalidePyTorchCudaHelpers.h>
#include <HalideBuffer.h>
#include <HalidePyTorchHelpers.h>

#include "KAS/CodeGen/Loader.hpp"
#include "KAS/Utils/Common.hpp"


namespace kas {

Loader::Loader(const std::string& path, const std::string& symbol, bool cuda, std::size_t countInputs, std::size_t countKernels):
    cuda { cuda },
    countInputs { countInputs }
{
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        KAS_CRITICAL("Failed to load dynamic library: {}", dlerror());
    }
    for (std::size_t i = 0; i < countKernels; ++i) {
        std::string forwardName = fmt::format("{}_{}", symbol, i);
        std::string backwardName = fmt::format("{}_{}_grad", symbol, i);

        void *forwardFnPtr = dlsym(handle, forwardName.c_str());
        void *backwardFnPtr = dlsym(handle, backwardName.c_str());
        if (!forwardFnPtr || !backwardFnPtr) {
            KAS_CRITICAL("Failed to load {}th kernel {}", i, symbol);
        }

        forwardPipelines.push_back(forwardFnPtr);
        backwardPipelines.push_back(backwardFnPtr);
    }
}

void Loader::call(const std::size_t expectedCountBuffers, void *pipeline, const std::vector<at::Tensor *>& buffers) const {
    if (cuda) {
        // Setup CUDA.
        int device_id = at::cuda::current_device();
        CUcontext ctx = 0;
        CUresult res = cuCtxGetCurrent(&ctx);
        KAS_ASSERT(res == 0, "Could not acquire CUDA context.");
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id);
        struct UserContext { int device_id; CUcontext *cuda_context; cudaStream_t *stream; } user_ctx;
        user_ctx.device_id = device_id;
        user_ctx.cuda_context = &ctx;
        user_ctx.stream = &stream;
        const void *__user_context = reinterpret_cast<const void *>(&user_ctx);

        std::vector<Halide::Runtime::Buffer<float>> wrapped;
        wrapped.reserve(expectedCountBuffers);
        // Check tensors have contiguous memory and are on the correct device.
        for (std::size_t i = 0; at::Tensor *buffer: buffers) {
            KAS_ASSERT(buffer->is_contiguous(), "Input tensor {} is not contiguous.", i);
            KAS_ASSERT(buffer->is_cuda() && buffer->device().index() == device_id, "Input tensor {} is not on device {}.", i, device_id);
            wrapped.push_back(Halide::PyTorch::wrap_cuda<float>(*buffer));
            ++i;
        }

        int err;
        using BufPtr = struct halide_buffer_t *;
        // Run Halide pipeline.
        switch (expectedCountBuffers) {
        case 0: err = reinterpret_cast<int (*)(const void *)>(pipeline)(__user_context); break;
        case 1: err = reinterpret_cast<int (*)(const void *, BufPtr)>(pipeline)(__user_context, wrapped[0]); break;
        case 2: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1]); break;
        case 3: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1], wrapped[2]); break;
        case 4: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1], wrapped[2], wrapped[3]); break;
        case 5: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4]); break;
        case 6: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5]); break;
        case 7: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6]); break;
        case 8: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6], wrapped[7]); break;
        case 9: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6], wrapped[7], wrapped[8]); break;
        case 10: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6], wrapped[7], wrapped[8], wrapped[9]); break;
        case 11: err = reinterpret_cast<int (*)(const void *, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(__user_context, wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6], wrapped[7], wrapped[8], wrapped[9], wrapped[10]); break;
        default: KAS_CRITICAL("Too many buffers!");
        }
        AT_ASSERTM(err == 0, "Error {} when executing Halide pipeline.", err);

        // Make sure data is on device
        for (std::size_t i = 0; auto& buffer: wrapped) {
            KAS_ASSERT(!buffer.host_dirty(), "Device not synchronized for buffer {}. Make sure all update stages are explicitly computed on GPU.", i);
            buffer.device_detach_native();
            ++i;
        }
    } else {
        std::vector<Halide::Runtime::Buffer<float>> wrapped;
        wrapped.reserve(expectedCountBuffers);
        // Check tensors have contiguous memory.
        for (std::size_t i = 0; at::Tensor *buffer: buffers) {
            KAS_ASSERT(buffer->is_contiguous(), "Input tensor {} is not contiguous.", i);
            wrapped.push_back(Halide::PyTorch::wrap<float>(*buffer));
            ++i;
        }

        int err;
        using BufPtr = struct halide_buffer_t *;
        // Run Halide pipeline.
        switch (expectedCountBuffers) {
        case 0: err = reinterpret_cast<int (*)()>(pipeline)(); break;
        case 1: err = reinterpret_cast<int (*)(BufPtr)>(pipeline)(wrapped[0]); break;
        case 2: err = reinterpret_cast<int (*)(BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1]); break;
        case 3: err = reinterpret_cast<int (*)(BufPtr, BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1], wrapped[2]); break;
        case 4: err = reinterpret_cast<int (*)(BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1], wrapped[2], wrapped[3]); break;
        case 5: err = reinterpret_cast<int (*)(BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4]); break;
        case 6: err = reinterpret_cast<int (*)(BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5]); break;
        case 7: err = reinterpret_cast<int (*)(BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6]); break;
        case 8: err = reinterpret_cast<int (*)(BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6], wrapped[7]); break;
        case 9: err = reinterpret_cast<int (*)(BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6], wrapped[7], wrapped[8]); break;
        case 10: err = reinterpret_cast<int (*)(BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6], wrapped[7], wrapped[8], wrapped[9]); break;
        case 11: err = reinterpret_cast<int (*)(BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr, BufPtr)>(pipeline)(wrapped[0], wrapped[1], wrapped[2], wrapped[3], wrapped[4], wrapped[5], wrapped[6], wrapped[7], wrapped[8], wrapped[9], wrapped[10]); break;
        default: KAS_CRITICAL("Too many buffers!");
        }
        AT_ASSERTM(err == 0, "Error {} when executing Halide pipeline.", err);
    }
}

void Loader::forward(std::size_t index, const std::vector<at::Tensor *>& buffers) const {
    const std::size_t expectedCountBuffers = countInputs + 1;
    KAS_ASSERT(buffers.size() == expectedCountBuffers, "Expected {} input tensors and 1 result tensor in forward pipeline, got {} buffers.", countInputs, buffers.size());
    void *pipeline = forwardPipelines.at(index);

    call(expectedCountBuffers, pipeline, buffers);
}

void Loader::backward(std::size_t index, const std::vector<at::Tensor *>& buffers) const {
    const std::size_t expectedCountBuffers = 2 * countInputs + 1;
    KAS_ASSERT(buffers.size() == expectedCountBuffers, "Expected {0} input tensors, 1 output gradient tensor and {0} gradient tensor in backward pipeline, got {1} buffers.", countInputs, buffers.size());
    void *pipeline = backwardPipelines.at(index);

    call(expectedCountBuffers, pipeline, buffers);
}

Loader::~Loader() {
    dlclose(handle);
}

} // namespace kas
