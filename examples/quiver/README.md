# Using Quiver for PyG Examples

**[Quiver](https://github.com/quiver-team/torch-quiver)** is a **GPU-optimized distributed library** for PyG.
It can speed up graph sampling and feature aggregation through GPU when running PyG examples.

## Installation

Assuming you have installed PyTorch and PyG, you can install Quiver as follows:

```bash
pip install torch-quiver>=0.1.1
```

## Usage

The API and design documentation of Quiver can be found [here](https://github.com/quiver-team/torch-quiver).

## Examples

We provide several examples to showcase the usage of Quiver within PyG:

### Single-GPU Training

The single-GPU example leverages Quiver's ability of **(i)** GPU-based graph sampling and feature aggregation, and **(ii)** GNN data caching algorithm (which cache hot data in GPU memory) while enabling fast access to CPU data using a Quiver shared tensor implementation:

```bash
python single_gpu_quiver.py
```

### Multi-GPU Training

The multi-GPU example further leverages Quiver's ability of **(i)** distributing sampling and feature aggregation to multiple GPUs, and **(ii)** using multi-GPU memories to cache and replicate hot GNN data:

```bash
python multi_gpu_quiver.py
```

### Distributed Training

A Quiver-based distributed PyG example is coming soon.
