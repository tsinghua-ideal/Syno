FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Install system dependencies
RUN mamba install -y \
    gcc=14 gxx=14 \
    cmake libpng zlib libzlib binutils \
    boost \
    nlohmann_json \
    gtest gmock \
    opencv cupy numba libjpeg-turbo libtiff pillow \
    pkg-config pybind11 tmux -c conda-forge && \
    mamba clean --all

COPY . /workspace

WORKDIR /workspace

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install thop timm tqdm datasets transformers multiset pytest torch_geometric torchmetrics ffcv \
    matplotlib seaborn easypyplot && \
    pip3 install . && \
    pip3 cache purge
