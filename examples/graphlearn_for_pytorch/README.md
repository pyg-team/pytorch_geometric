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


- [Installation](#installation)
  - [Requirements](#requirements)
  - [Pip Wheels](#pip-wheels)
- [Examples](#examples)
  - [Accelarating PyG model training on a single GPU.](#accelarating-pyg-model-training-on-a-single-gpu)
  - [Distributed training](#distributed-training)

## Installation

### Requirements
- cuda
- python>=3.6
- torch(PyTorch)
- torch_geometric, torch_scatter, torch_sparse. Please refer to [PyG](https://github.com/pyg-team/pytorch_geometric) for installation.
### Pip Wheels

```
# glibc>=2.14, torch>=1.13
pip install graphlearn-torch
```

## Examples

### Accelarating PyG model training on a single GPU.

First, let's take PyG's [GraphSAGE on OGBN-Products](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py)
as an example. To benefit from the acceleration of model training by using GLT, you only need to replace PyG's `torch_geometric.loader.NeighborLoader`
by the `graphlearn_torch.loader.NeighborLoader`
.

```python
import torch
import graphlearn_torch as glt
import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset

# PyG's original code preparing the ogbn-products dataset
root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
data = dataset[0]

# Enable GLT acceleration on PyG requires only replacing
# PyG's NeighborSampler with the following code.
glt_dataset = glt.data.Dataset()
glt_dataset.build(edge_index=data.edge_index,
                  feature_data=data.x,
                  sort_func=glt.data.sort_by_in_degree,
                  split_ratio=0.2,
                  label=data.y,
                  device=0)
train_loader = glt.loader.NeighborLoader(glt_dataset,
                                         [15, 10, 5],
                                         split_idx['train'],
                                         batch_size=1024,
                                         shuffle=True,
                                         drop_last=True,
                                         as_pyg_v1=True)
```

The complete example can be found in the GLT example [`train_sage_ogbn_products.py`](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/train_sage_ogbn_products.py).

### Distributed training

Generally speaking, for distributed training, steps can be as follows:

1. Load the graphs and features from partitions.

2. Create the distributed neighbor loader based on the dataset above.

3. Define a PyTorch DDP model and run.

For distributed training, we have implemented multi-processing asynchronous sampling,
pin memory buffer, hot feature cache, and applied fast networking
technologies (PyTorch RPC with RDMA support) to speed up distributed sampling
and reduce communication. As a result, GLT can achieve high
scalability and support graphs with billions of edges.

Further information for the GLT distributed example can be found in the [distributed training example doc](distributed/README.md).
