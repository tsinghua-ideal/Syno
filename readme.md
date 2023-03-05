# Kernel Architechture Search

## Build Dependencies

- A C++20-compatible compiler and CMake. Ninja is recommended.
- [Halide with GPU Autoscheduler](https://github.com/aekul/Halide/tree/gpu-autoscheduler), with [this patch](./bugfix.patch) applied. For more information, refer to the [pull request](https://github.com/halide/Halide/pull/6856).
- [PyTorch](https://github.com/pytorch/pytorch).
- [Boost](https://github.com/boostorg/boost).
- [fmtlib](https://github.com/fmtlib/fmt).
- [pybind11](https://github.com/pybind/pybind11).
- [GoogleTest](https://github.com/google/googletest).

## Build and Run

CMake tests are available.

```bash
mkdir build && cd build
cmake -G Ninja ..
cmake --build .
ctest
```

This project uses [`scikit-build`](https://github.com/scikit-build/scikit-build-core), so installing the Python module is fairly simple.

```bash
pip install .
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
