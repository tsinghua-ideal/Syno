# Kernel Architechture Search

## Build Dependencies

- A C++20-compatible compiler and CMake. Ninja is recommended.
- Trunk version of [Halide](https://github.com/halide/Halide), due to API changes to auto schedulers.
- [PyTorch](https://github.com/pytorch/pytorch).
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
