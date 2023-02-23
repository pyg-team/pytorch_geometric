FROM ubuntu:18.04

# metainformation
LABEL org.opencontainers.image.version = "2.1.0"
LABEL org.opencontainers.image.authors = "Matthias Fey"
LABEL org.opencontainers.image.source = "https://github.com/pyg-team/pytorch_geometric"
LABEL org.opencontainers.image.licenses = "MIT"
LABEL org.opencontainers.image.base.name="docker.io/library/ubuntu:18.04"

RUN apt-get update && apt-get install -y apt-transport-https ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils gnupg2 curl && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list &&\
    apt-get purge --autoremove -y curl && \
rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243
ENV NCCL_VERSION 2.4.8
ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1
ENV CUDNN_VERSION 7.6.5.32

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-1 && \
    ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libcublas10=10.2.1.243-1 \
        libnccl2=$NCCL_VERSION-1+cuda10.1 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.1 \
        libcublas-dev=10.2.1.243-1 \
        && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 \
    && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# NVIDIA docker 1.0.
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# NVIDIA container runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"

# PyTorch (Geometric) installation
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update &&  apt-get install -y \
    curl \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda.
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment.
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 10.0-specific steps.
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    "pytorch=1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0" \
    torchvision=0.5.0=py36_cu101 \
 && conda clean -ya

# Install HDF5 Python bindings.
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install TorchNet, a high-level framework for PyTorch.
RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests.
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz.
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
 && conda clean -ya

# Install OpenCV3 Python bindings.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya

# Install PyG.
RUN CPATH=/usr/local/cuda/include:$CPATH \
 && LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
 && DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN pip install scipy

RUN pip install --no-index torch_scatter -f https://data.pyg.org/whl/torch-1.4.0+cu101.html \
 && pip install --no-index torch_sparse -f https://data.pyg.org/whl/torch-1.4.0+cu101.html \
 && pip install --no-index torch_cluster -f https://data.pyg.org/whl/torch-1.4.0+cu101.html \
 && pip install --no-index torch_spline_conv -f https://data.pyg.org/whl/torch-1.4.0+cu101.html \
 && pip install torch-geometric

# Set the default command to python3.
CMD ["python3"]
