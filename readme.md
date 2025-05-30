# Syno: Structured Synthesis for Neural Operators

This is the source code repository of the ASPLOS '25 paper "Syno: Structured Synthesis for Neural Operators".

We propose a novel end-to-end framework, Syno, for optimizing the inference performance of neural networks by synthesizing novel operators from scratch.

## Reproducing Our Results

Please see the instructions in the [Artifact Evaluation Repository](https://github.com/Yongqi-Zhuo/Syno-AE) for detailed steps on how to replicate our experiments in the paper.

## Use Cases

Define a neural network with some operators being `Placeholder`s, which are slots to be filled by Syno. For example, Syno can synthesize a convolution operator from scratch: we do not hardcode a convolution within Syno!

Note that this can also be done automatically, by substituting certain type of operator in a given neural network.

```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # `Placeholder`s can provide concrete values for symbolic variables. This can be used to match spatial dimensions of tensors.
        self.kernel_1 = Placeholder({"H": 32, "W": 32})
        self.kernel_2 = Placeholder({"H": 16, "W": 16})
        self.kernel_3 = Placeholder({"H": 16, "W": 16})

    def forward(self, x: torch.Tensor):
        x = self.kernel_1(x)
        x = x.view(1, 1, 32, 32)
        x = F.avg_pool2d(x, 2)
        x = x.view(16, 16)
        x = self.kernel_2(x)
        x = self.kernel_3(x)
        return x
```

Then, obtain a `Sampler` to start synthesizing operators.

```python
# Obtain a model with `Placeholder`s inside
net = Model()
# Obtain a sampler that handles operator synthesis for you
sampler = Sampler(
    "[H, W]", # Symbolic input shape of the operator
    "[H, W]", # Symbolic output shape of the operator
    ["H: 2", "W: 2"], # Some hyperparameters for how many times a symbolic variable can be used
    ["s_1=2: 2", "s_2=3: 2"], # Some coefficients to be used as auxiliary symbolic variables
    net=net, # Pass our model in, and `Placeholder`s are analyzed.
)
# Synthesize a novel operator to be substituted into the model
operator = sampler.sample()
# Code generation
kernel_packs = operator.construct_kernel_packs()
# Replace the `Placeholder`s in the model with the novel operators we have just synthesized
sampler.replace(net, kernel_packs)
```

We have observed novel operators beyond current knowledge! Please see our paper for more information.

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

### Alternative: Using Docker

We have provided a Dockerfile for environment setup.

```bash
docker build -t syno .
docker run -it --gpus all --shm-size=16.00gb syno
```

## Scripts for Experiments

Experiments' codes for searching kernels on MNIST and CIFAR10 are provided under the `experiments` folder. To launch an experiment using 8 GPUs, run

```bash
bash ./run_server.sh $SEARCH_ITERATIONS
bash ./run_tmux.sh 8 python -u evaluator.py
```

## FAQ

### How to debug

Uncomment the line `set(CMAKE_BUILD_TYPE Debug)` in `CMakeLists.txt`, and rebuild.

## Cite

```
@inproceedings{zhuo2025syno,
  title={Syno: Structured Synthesis for Neural Operators},
  author={Zhuo, Yongqi and Su, Zhengyuan and Zhao, Chenggang and Gao, Mingyu},
  booktitle={Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
  pages={to appear},
  year={2025}
}
```

