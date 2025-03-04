# Examples

This folder contains a plethora of examples covering different GNN use-cases.
This readme highlights some key examples.

A great and simple example to start with is [`gcn.py`](./gcn.py), showing a user how to train a [`GCN`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html) model for node-level prediction on small-scale homogeneous data.

For a simple link prediction example, see [`link_pred.py`](./link_pred.py).

For examples on [Open Graph Benchmark](https://ogb.stanford.edu/) datasets, see the `ogbn_*.py` examples:

- [`ogbn_train.py`](./ogbn_train.py) is an example for training a GNN on the large-scale `ogbn-papers100m` dataset, containing approximately ~1.6B edges or the medium scale `ogbn-products` dataset, ~62M edges.
  - Uses SGFormer (a kind of GraphTransformer) by default.
  - [SGFormer Paper](https://arxiv.org/pdf/2306.10759)
- [`ogbn_proteins_deepgcn.py`](./ogbn_proteins_deepgcn.py) is an example to showcase how to train deep GNNs on the `ogbn-proteins` dataset.
- [`ogbn_train_cugraph.py`](./ogbn_train_cugraph.py) shows how to accelerate the `ogbn_train.py` workflow using [CuGraph](https://github.com/rapidsai/cugraph).

For examples on using `torch.compile`, see the examples under [`examples/compile`](./compile).

For examples on scaling PyG up via multi-GPUs, see the examples under [`examples/multi_gpu`](./multi_gpu).

For examples on working with heterogeneous data, see the examples under [`examples/hetero`](./hetero).

For examples on co-training LLMs with GNNs, see the examples under [`examples/llm`](./llm).
