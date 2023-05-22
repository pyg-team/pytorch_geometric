# Using GraphLearn-for-PyTorch(GLT) for Distributed Acceleration for PyG

**[GraphLearn-for-PyTorch(GLT)](https://github.com/alibaba/graphlearn-for-pytorch)** is a graph learning library for PyTorch that makes
distributed GNN training and inference easy and efficient. It leverages the
power of GPUs to accelerate graph sampling and utilizes UVA to reduce the
conversion and copying of features of vertices and edges. For large-scale graphs,
it supports distributed training on multiple GPUs or multiple machines through
fast distributed sampling and feature lookup. Additionally, it provides flexible
deployment for distributed training to meet different requirements.

Most of the APIs of GLT are compatible with PyG,
so you only need to modify a few lines of PyG's code to get the
acceleration of the program. For GLT-specific APIs,
they are also compatible with PyTorch and there are complete documentations
and usage examples available.


## Installation

### Requirements
- cuda>=11.4
- python>=3.6
- torch(PyTorch)>=1.13
- torch_geometric, torch_scatter, torch_sparse. Please refer to [PyG](https://github.com/pyg-team/pytorch_geometric) for installation.
### Pip Wheels

```
# glibc>=2.14, torch>=1.13
pip install graphlearn-torch
```

## Examples

### Accelarating PyG on a single node.
**Single-GPU:**

Let's take PyG's [GraphSAGE on OGBN-Products](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py)
as an example. To benefit from the acceleration by using GLT, you only need to replace PyG's `torch_geometric.loader.NeighborLoader`
by the `graphlearn_torch.loader.NeighborLoader`, which is .

The complete example can be found in the GLT example [`train_sage_ogbn_products.py`](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/train_sage_ogbn_products.py).

**Multi-GPU**

The multi-GPU example further leverages GLT's ability of using [`UnifiedTensor`](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/docs/tutorial/basic_object.md?plain=1#L97-L112) to cache hot feature data in multi-GPU memories, and
introducing [`DeviceGroup`](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/docs/tutorial/basic_object.md?plain=1#L142-L162) in GLT significantly expands the cache capacity of
feature data in GPU and make full utilization of the NVLink bandwidth.

The example can be found in the GLT example [`train_sage_ogbn_papers100m.py`](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/multi_gpu/train_sage_ogbn_papers100m.py).

### Distributed training

Generally speaking, for distributed training, steps can be as follows:

1. Setup distributed environment and do partitioning on the dataset.

2. Load the graphs and features from partitions.

3. Create the distributed neighbor loader based on the dataset above.

4. Define a PyTorch DDP model and run.

For distributed training, we have implemented multi-processing asynchronous sampling,
pin memory buffer, hot feature cache, and applied fast networking
technologies (PyTorch RPC with RDMA support) to speed up distributed sampling
and reduce communication. As a result, GLT can achieve high
scalability and support graphs with billions of edges.

Further information for the GLT distributed example can be found in the [distributed training example doc](distributed/README.md).
