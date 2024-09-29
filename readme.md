# Kernel Architechture Search

## Build Dependencies

- A C++20-compatible compiler. (Complete support required, which means, GCC >= 12 if you are using GCC.)
- CMake.
- Ninja.
- [Halide with GPU Autoscheduler](https://github.com/aekul/Halide/tree/gpu-autoscheduler), with [this patch](./bugfix.patch) applied. For more information, refer to the [pull request](https://github.com/halide/Halide/pull/6856).
- CUDA.
- [PyTorch](https://github.com/pytorch/pytorch).
- [Boost](https://github.com/boostorg/boost).
- [nlohmann-json](https://github.com/nlohmann/json).
- [fmtlib](https://github.com/fmtlib/fmt).
- [pybind11](https://github.com/pybind/pybind11).
- [GoogleTest](https://github.com/google/googletest).

Note: If you are using anaconda to manage your packages, you may also install the dependency via 

```[language=bash]
mamba create -n kas python=3.10 gcc=12 gxx=12 cmake ninja zlib libpng libzlib ncurses -c conda-forge
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install boost nlohmann_json fmt=10 pybind11 gtest gmock pytest -c conda-forge
pip install -r requirements.txt
```

HACK: comment line 130-134 in `$CONDA_PATH/envs/kas/include/crt/host_config.h`

## Build and Run

CMake tests are available.

```bash
cmake -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=true -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` .
cmake --build build --parallel
cd build && ctest
```

This project uses [`scikit-build`](https://github.com/scikit-build/scikit-build-core), so installing the Python module is fairly simple.
Note: please prepare at least 10GB RAM. 

```bash
pip install . --config-settings=cmake.define.CMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
pytest
```

See all the tests in `tests/`.

### Disabling Halide

Halide has been made optional. If you do not want to use Halide codegen, change the line `option(KAS_WITH_HALIDE "..." ON)` in `CMakeLists.txt` to `OFF`. This disables all the tests as well.

## Scripts for Experiments

Experiments' codes for searching kernels on MNIST and CIFAR10 are provided under the `experiments` folder. To launch an experiment using 8 GPUs, run

```bash
bash ./run_server.sh $SEARCH_ITERATIONS
bash ./run_tmux.sh 8 python -u evaluator.py
```

## Notes

To run `tests/Semantics/test_semantics_*.py`, you need to first run `ctest` to generate the kernels. Then you can either manually run it to observe the performance or just run `pytest` to check correctness.

## FAQ

### When configuring with CMake, CUDA runtime complains "unsupported GNU version! gcc versions later than 11 are not supported!"

Since this project strictly requires full support of C++20, you need to use GCC 12. However, for the time being, nvcc does not support GCC 12. Fortunately, we are not using nvcc to compile the kernels, and we only need to bypass this restriction by modifying `host_config.h` (which is usually located at `/usr/include/crt/host_config.h`) of CUDA runtime, removing the check that prevents us from using GCC 12.

### How to debug

Uncomment the line `set(CMAKE_BUILD_TYPE Debug)` in `CMakeLists.txt`, and rebuild.
