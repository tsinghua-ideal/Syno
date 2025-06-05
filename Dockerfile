ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3

FROM ${BASE_IMAGE}

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake \
        libpng-dev \
        zlib1g-dev \
        binutils \
        libboost-all-dev \
        libfmt-dev \
        nlohmann-json3-dev \
        libgtest-dev \
        libgmock-dev \
        libopencv-dev \
        python3-opencv \
        libjpeg-dev \
        libjpeg-turbo8-dev \
        libturbojpeg0-dev \
        libtiff-dev \
        python3-pil \
        pkg-config \
        pybind11-dev \
        llvm-dev \
        tmux \
        rsync && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sv /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Fix: /opt/hpcx/ucc/lib/libucc.so.1: undefined symbol: ucs_config_doc_nop
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/hpcx/ucx/lib

WORKDIR /workspace

# Copy TVM configuration files
COPY ./experiments/performance/external/tvm.* /workspace/

# Install TVM
# Note: add /usr/local/cuda/lib64/stubs to LD_LIBRARY_PATH to fix: undefined reference to `cuLaunchKernel`
RUN \
    git clone --recursive https://github.com/apache/tvm.git && \
    pushd tvm && \
    git checkout $(cat ../tvm.hash) && \
    patch -p1 < ../tvm.patch && \
    mkdir build && \
    cp ../tvm.cmake build/config.cmake && \
    pushd build && \
    cmake .. && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs && \
    cmake --build . --parallel $(nproc) --target install && \
    popd && \
    export TVM_LIBRARY_PATH=/usr/local/lib && \
    pip install -vvv ./python && \
    popd && \
    rm -rf tvm tvm.*

# Copy PyTorch patch
COPY ./experiments/performance/external/torch.patch /workspace/

# Patch PyTorch
RUN \
    patch /usr/local/lib/python3.12/dist-packages/torch/_dynamo/symbolic_convert.py -p5 < /workspace/torch.patch && \
    rm /workspace/torch.patch

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install \
        pandas==2.2.3 \
        cupy-cuda12x \
        numba \
        thop \
        timm \
        tqdm \
        datasets \
        transformers \
        multiset \
        pytest \
        torch_geometric \
        torchmetrics \
        ffcv \
        tornado \
        xgboost \
        cloudpickle \
        psutil \
        matplotlib \
        seaborn \
        easypyplot && \
    pip3 cache purge

COPY . /workspace/Syno

RUN pip3 install ./Syno && \
    pip3 cache purge
