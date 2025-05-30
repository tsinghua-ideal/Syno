# Syno: Structured Synthesis for Neural Operators

This is the source code repository of the ASPLOS '25 paper "Syno: Structured Synthesis for Neural Operators".

## Build Dependencies

- A C++20-compatible compiler. (Complete support required, which means, GCC >= 12 if you are using GCC.)
- CMake.
- Ninja.
- CUDA.
- [PyTorch](https://github.com/pytorch/pytorch).
- [Boost](https://github.com/boostorg/boost).
- [nlohmann-json](https://github.com/nlohmann/json).
- [fmtlib](https://github.com/fmtlib/fmt).
- [pybind11](https://github.com/pybind/pybind11).
- [GoogleTest](https://github.com/google/googletest).

Note: If you are using anaconda to manage your packages, you may also install the dependency via 

```[language=bash]
mamba create -n kas python=3.12 gcc=14 gxx=14 clang=19 clangdev llvm llvm-openmp=19 llvmdev lld lit python-clang -c conda-forge
mamba install cmake ninja -c conda-forge
mamba install zlib libpng libzlib ncurses -c conda-forge
mamba install cudatoolkit=12.9 cudnn=9.10.1 -c conda-forge
mamba install cudatoolkit-dev=12.9 -c conda-forge
mamba install pytorch torchvision torchaudio pytorch-cuda=12.9 -c pytorch -c nvidia
mamba install boost nlohmann_json fmt pybind11 gtest gmock pytest -c conda-forge
mamba install cupy pkg-config libjpeg-turbo opencv -c conda-forge # To enable ffcv-imagenet
pip install -r requirements.txt
```

## Build and Run

CMake tests are available.

```bash
cmake -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=true` .
cmake --build build
cd build && ctest
```

This project uses [`scikit-build`](https://github.com/scikit-build/scikit-build-core), so installing the Python module is fairly simple.

```bash
pip install .`
pytest
```

See all the tests in `tests/`.

## Scripts for Experiments

Experiments' codes for searching kernels on MNIST and CIFAR10 are provided under the `experiments` folder. To launch an experiment using 8 GPUs, run

```bash
bash ./run_server.sh $SEARCH_ITERATIONS
bash ./run_tmux.sh 8 python -u evaluator.py
```

## FAQ

### How to debug

Uncomment the line `set(CMAKE_BUILD_TYPE Debug)` in `CMakeLists.txt`, and rebuild.
