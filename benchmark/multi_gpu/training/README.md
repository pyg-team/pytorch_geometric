# Training Benchmark

## Running benchmark on CUDA GPU

Run benchmark, e.g. assuming you have `n` NVIDIA GPUs:

```
python training_benchmark_cuda.py --dataset ogbn-products --model edge_cnn --num-epochs 3 --n_gpus <n>
```

## Running benchmark on Intel GPU

### Environment setup

### Prerequisites

- Intel Data Center GPU Max Series. You could try it through [Intel DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/services.html).
- Verify the Intel GPU Driver is installed, refer to the [guide](https://dgpu-docs.intel.com/driver/installation.html).

### docker setup

If you want to run your scripts inside a docker image, you could refer to the [dockerfile](https://github.com/pyg-team/pytorch_geometric/blob/master/docker/Dockerfile.xpu) and the corresponding [guide](https://github.com/pyg-team/pytorch_geometric/blob/master/docker).

### bare-metal setup

If you prefer to run your scripts directly on the bare-metal server. We recommend the installation guidance provided by [Intel® Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.30%2bxpu&os=linux%2fwsl2&package=pip). The following are some key steps:

- Install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html), indluding [Intel® oneAPI DPC++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html), [Intel® oneAPI Math Kernel Library (oneMKL)](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2024-1/intel-oneapi-math-kernel-library-onemkl.html), [Intel® oneAPI Collective Communications Library (oneCCL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html), and [Intel® oneCCL Bindings for PyTorch](https://github.com/intel/torch-ccl).

```bash
# Install oneCCL package on Ubuntu
sudo apt install -y intel-oneapi-dpcpp-cpp-2024.1=2024.1.0-963 intel-oneapi-mkl-devel=2024.1.0-691 intel-oneapi-ccl-devel=2021.12.0-309
# Install oneccl_bindings_for_pytorch
pip install oneccl_bind_pt==2.1.300+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# Runtime Dynamic Linking
source /opt/intel/oneapi/setvars.sh
```

- Install [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) and the corresponding version of PyTorch

```bash
pip install torch==2.1.0.post2 intel-extension-for-pytorch==2.1.30+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### Running benchmark

This [guide](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/features/DDP.html) is helpful for you to launch DDP training on intel GPU.

To Run benchmark, e.g. assuming you have `n` XPUs:

```
mpirun -np <n> python training_benchmark_xpu.py --dataset ogbn-products --model edge_cnn --num-epochs 3
```
