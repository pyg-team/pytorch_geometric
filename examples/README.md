# Examples

This folder contains a plethora of examples covering different GNN use-cases.
This readme highlights some key examples.

A great and simple example to start with is [`planetoid_train.py`](./planetoid_train.py), showing a user how to train a [`GCN`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html) model for node-level prediction on small-scale homogeneous data.

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

# Results

```
python examples/planetoid_train.py --gnn_choice {model} --dataset {dataset}
```

| Model     | Cora       | CiteSeer   | PubMed     |
| --------- | ---------- | ---------- | ---------- |
| GCN       | 83.80%     | 72.40%     | 87.20%     |
| GAT       | 91.40%     | 76.00%     | 84.20%     |
| AGNN      | 92.00%     | 77.20%     | 87.80%     |
| ARMA      | 88.60%     | 74.40%     | 86.60%     |
| MixHop    | 91.40%     | 75.80%     | 84.60%     |
| SGC       | 64.00%     | 73.60%     | 77.00%     |
| TAGCN     | 91.20%     | 73.80%     | **89.00%** |
| GraphUNet | 90.20%     | 77.20%     | 85.20%     |
| PMLP      | 90.40%     | **77.40%** | 84.60%     |
| SplineGNN | 91.60%     | **77.40%** | 87.60%     |
| GCN2      | **92.80%** | 75.60%     | 84.60%     |
| SuperGAT  | 88.20%     | 74.40%     | 80.60%     |
| DNA       | 90.00%     | 75.20%     | 85.80%     |
| Node2Vec  | 84.00%     | 55.60%     | 76.80%     |
