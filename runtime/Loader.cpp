#include <array>

#include <dlfcn.h>
#include <fmt/core.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <HalidePyTorchCudaHelpers.h>
#include <HalideBuffer.h>
#include <HalidePyTorchHelpers.h>

#include "Loader.hpp"

#include "KAS/Utils/Common.hpp"
#include "KAS/Utils/VarArg.hpp"


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

namespace {

using BufPtr = struct halide_buffer_t *;

template<std::size_t ExpectedCountBuffers, std::size_t... Is>
int CallWithCtx(void *pipeline, const void *__user_context, std::array<Halide::Runtime::Buffer<float>, ExpectedCountBuffers>& wrapped, std::index_sequence<Is...>) {
    using Signature = FunctionPtrOfNArgs<int, BufPtr, ExpectedCountBuffers, const void *>;
    return reinterpret_cast<Signature>(pipeline)(__user_context, wrapped[Is]...);
}

template<std::size_t ExpectedCountBuffers, std::size_t... Is>
int CallWithoutCtx(void *pipeline, std::array<Halide::Runtime::Buffer<float>, ExpectedCountBuffers>& wrapped, std::index_sequence<Is...>) {
    using Signature = FunctionPtrOfNArgs<int, BufPtr, ExpectedCountBuffers>;
    return reinterpret_cast<Signature>(pipeline)(wrapped[Is]...);
}

template<std::size_t ExpectedCountBuffers>
void LoaderCallImpl(bool cuda, void *pipeline, const std::vector<at::Tensor *>& buffers) {
    std::array<Halide::Runtime::Buffer<float>, ExpectedCountBuffers> wrapped;
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

        // Check tensors have contiguous memory and are on the correct device.
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            at::Tensor *buffer = buffers[i];
            KAS_ASSERT(buffer->is_contiguous(), "Input tensor {} is not contiguous.", i);
            KAS_ASSERT(buffer->is_cuda() && buffer->device().index() == device_id, "Input tensor {} is not on device {}.", i, device_id);
            wrapped[i] = Halide::PyTorch::wrap_cuda<float>(*buffer);
        }

        // Run Halide pipeline.
        int err = CallWithCtx<ExpectedCountBuffers>(pipeline, __user_context, wrapped, std::make_index_sequence<ExpectedCountBuffers>());
        KAS_ASSERT(err == 0, "Error {} when executing Halide pipeline.", err);

        // Make sure data is on device
        for (std::size_t i = 0; i < wrapped.size(); ++i) {
            auto& buffer = wrapped[i];
            KAS_ASSERT(!buffer.host_dirty(), "Device not synchronized for buffer {}. Make sure all update stages are explicitly computed on GPU.", i);
            buffer.device_detach_native();
        }
    } else {
        // Check tensors have contiguous memory.
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            at::Tensor *buffer = buffers[i];
            KAS_ASSERT(buffer->is_contiguous(), "Input tensor {} is not contiguous.", i);
            wrapped[i] = Halide::PyTorch::wrap<float>(*buffer);
        }

        // Run Halide pipeline.
        int err = CallWithoutCtx<ExpectedCountBuffers>(pipeline, wrapped, std::make_index_sequence<ExpectedCountBuffers>());
        KAS_ASSERT(err == 0, "Error {} when executing Halide pipeline.", err);
    }
}

} // namespace

void Loader::call(const std::size_t expectedCountBuffers, void *pipeline, const std::vector<at::Tensor *>& buffers) const {
    switch (expectedCountBuffers) {
    case 0: LoaderCallImpl<0>(cuda, pipeline, buffers); break;
    case 1: LoaderCallImpl<1>(cuda, pipeline, buffers); break;
    case 2: LoaderCallImpl<2>(cuda, pipeline, buffers); break;
    case 3: LoaderCallImpl<3>(cuda, pipeline, buffers); break;
    case 4: LoaderCallImpl<4>(cuda, pipeline, buffers); break;
    case 5: LoaderCallImpl<5>(cuda, pipeline, buffers); break;
    case 6: LoaderCallImpl<6>(cuda, pipeline, buffers); break;
    case 7: LoaderCallImpl<7>(cuda, pipeline, buffers); break;
    case 8: LoaderCallImpl<8>(cuda, pipeline, buffers); break;
    case 9: LoaderCallImpl<9>(cuda, pipeline, buffers); break;
    case 10: LoaderCallImpl<10>(cuda, pipeline, buffers); break;
    case 11: LoaderCallImpl<11>(cuda, pipeline, buffers); break;
    case 12: LoaderCallImpl<12>(cuda, pipeline, buffers); break;
    case 13: LoaderCallImpl<13>(cuda, pipeline, buffers); break;
    case 14: LoaderCallImpl<14>(cuda, pipeline, buffers); break;
    case 15: LoaderCallImpl<15>(cuda, pipeline, buffers); break;
    case 16: LoaderCallImpl<16>(cuda, pipeline, buffers); break;
    case 17: LoaderCallImpl<17>(cuda, pipeline, buffers); break;
    }
}

void Loader::forward(std::size_t index, const std::vector<at::Tensor *>& buffers) const {
    const std::size_t ExpectedCountBuffers = countInputs + 1;
    KAS_ASSERT(buffers.size() == ExpectedCountBuffers, "Expected {} input tensors and 1 result tensor in forward pipeline, got {} buffers.", countInputs, buffers.size());
    void *pipeline = forwardPipelines.at(index);

    call(ExpectedCountBuffers, pipeline, buffers);
}

void Loader::backward(std::size_t index, const std::vector<at::Tensor *>& buffers) const {
    const std::size_t ExpectedCountBuffers = 2 * countInputs + 1;
    KAS_ASSERT(buffers.size() == ExpectedCountBuffers, "Expected {0} input tensors, 1 output gradient tensor and {0} gradient tensor in backward pipeline, got {1} buffers.", countInputs, buffers.size());
    void *pipeline = backwardPipelines.at(index);

    call(ExpectedCountBuffers, pipeline, buffers);
}

Loader::~Loader() {
    dlclose(handle);
}

} // namespace kas
