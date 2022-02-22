# Using Quiver for PyG Examples

[Quiver](https://github.com/quiver-team/torch-quiver) is a GPU-optimized distributed library for PyG. It can speed up graph sampling and feature aggregation through GPU when running PyG examples.

## Install

Assuming you have installed PyTorch and PyG, you can install Quiver as follow:

```bash
pip install torch-quiver==0.1.1
```

## Usage

The API and design documentation of Quiver can be found [here](https://github.com/quiver-team/torch-quiver). 

## Examples

We provide several examples to show how to use Quiver:

### 1-GPU PyG example

A 1-GPU PyG example can benefit from Quiver's ability of (i) GPU-based graph sampling and feature aggregation, and (ii) GNN data caching algorithm (which cache hot data in GPU memory) while enabling fast access to CPU data using a Quiver unified tensor implementation. To run this example, you can:

```bash
python .........
```

### Multi-GPU PyG example

PyG can further benefit from Quiver's ability of (i) distributing sampling and feature aggregation to multiple GPUs, and (ii) use multi-GPU memories to cache and replicate hot GNN data.

To run this example, you can:

```bash
python .........
```

### Distributed PyG example

A Quiver-based distributed PyG example is coming soon.