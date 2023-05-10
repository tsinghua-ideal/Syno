# Kernel Architechture Search

## Build Dependencies

- A C++20-compatible compiler.
- CMake.
- Ninja.
- [Halide with GPU Autoscheduler](https://github.com/aekul/Halide/tree/gpu-autoscheduler), with [this patch](./bugfix.patch) applied. For more information, refer to the [pull request](https://github.com/halide/Halide/pull/6856).
- CUDA.
- [PyTorch](https://github.com/pytorch/pytorch).
- [Boost](https://github.com/boostorg/boost).
- [fmtlib](https://github.com/fmtlib/fmt).
- [pybind11](https://github.com/pybind/pybind11).
- [GoogleTest](https://github.com/google/googletest).

Note: If you are using anaconda to manage your packages, you may also install the dependency via 

```[language=bash]
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install boost fmt pybind11 gtest gmock -c conda-forge
conda install cudatoolkit=11.7 cudnn=8.8.0
```

## Build and Run

CMake tests are available.

```bash
cmake -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=true -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` .
cmake --build build
ctest --test-dir build
```

This project uses [`scikit-build`](https://github.com/scikit-build/scikit-build-core), so installing the Python module is fairly simple.

```bash
pip install . --config-settings=cmake.define.CMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
pytest
```

See all the tests in `tests/`.

## Notes

To run `tests/semantics_test_*.py`, you need to first run `ctest` to generate the kernels. Then you can just

```bash
cd tests
python semantics_test_pool2d.py
python semantics_test_conv2d.py
```
