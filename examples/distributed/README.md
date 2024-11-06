# Examples for Distributed Graph Learning

This directory contains examples for distributed graph learning.
The examples are organized into two subdirectories:

1. [`NVIDIA-RAPIDS`](./NVIDIA-RAPIDS): Examples for scaling GNN training to any number of machines using NVIDIA RAPIDS.
1. [`pyg`](./pyg): Distributed training via PyG's own `torch_geometric.distributed` package by Intel
1. [`graphlearn_for_pytorch`](./graphlearn_for_pytorch): Distributed training via the external [GraphLearn-for-PyTorch (GLT)](https://github.com/alibaba/graphlearn-for-pytorch) package.
1. [`kuzu`](./kuzu): Remote backend via the [Kùzu](https://kuzudb.com/) graph database.
