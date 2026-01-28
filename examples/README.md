# Examples

This folder contains a plethora of examples covering different GNN use-cases.
This readme highlights some key examples.

Note: We recommend the [NVIDIA PyG Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pyg/tags) for best results and easiest setup with NVIDIA GPUs.

A great and simple example to start with is [`gcn.py`](./gcn.py), showing a user how to train a [`GCN`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html) model for node-level prediction on small-scale homogeneous data.

For a simple GNN based link prediction example, see [`link_pred.py`](./link_pred.py).

For an improved GNN based link prediction approach using Attract-Repel embeddings that can significantly boost accuracy (up to 23% improvement in AUC), see [`ar_link_pred.py`](./ar_link_pred.py). This approach is based on [Pseudo-Euclidean Attract-Repel Embeddings for Undirected Graphs](https://arxiv.org/abs/2106.09671).

To see an example for doing link prediction with an advanced Graph Transformer called [`LPFormer`](https://arxiv.org/abs/2310.11009), see \[`lpformer.py`\].

- [`ogbn_train.py`](./ogbn_train.py) is an example for training a GNN on the large-scale `ogbn-papers100m` dataset, containing approximately ~1.6B edges or the medium scale `ogbn-products` dataset, ~62M edges.
  - Uses SGFormer (a kind of GraphTransformer) by default.
  - [SGFormer Paper](https://arxiv.org/pdf/2306.10759)
  - [Polynormer](https://arxiv.org/pdf/2403.01232)
  - [Kumo.ai x NVIDIA x Stanford Graph Transformer Webinar](https://www.youtube.com/watch?v=wAYryx3GjLw)
- [`ogbn_proteins_deepgcn.py`](./ogbn_proteins_deepgcn.py) is an example to showcase how to train deep GNNs on the `ogbn-proteins` dataset.
- [`ogbn_train_cugraph.py`](./ogbn_train_cugraph.py) shows how to accelerate the `ogbn_train.py` workflow using [CuGraph](https://github.com/rapidsai/cugraph).
- [`ogbn_train_perforatedai.py`](https://github.com/PerforatedAI/PerforatedAI-Examples/tree/master/otherExamples/torch_geometric/OGBNProducts) shows how to optimize the `ogbn_train.py` workflow using [Perforated AI](https://github.com/PerforatedAI/PerforatedAI-API). Perforated AI provides a PyTorch add-on which increases network accuracy by empowering each artificial neuron with artificial dendrites.

For an example on [Relational Deep Learning](https://arxiv.org/abs/2312.04615) with the [RelBench datasets](https://relbench.stanford.edu/), see [`rdl.py`](./rdl.py).

For examples on using `torch.compile`, see the examples under [`examples/compile`](./compile).

For examples on working with heterogeneous data, see the examples under [`examples/hetero`](./hetero).

For examples on co-training LLMs with GNNs, see the examples under [`examples/llm`](./llm).

- [Stanford GNN+LLM Talk](https://www.nvidia.com/en-us/on-demand/session/other25-nv-0003/)

We recommend looking into [PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) for examples on setting up model parralel GNNs.

### Scale to Trillions of Edges with cuGraph

[cuGraph](https://github.com/rapidsai/cugraph) is a collection of packages focused on GPU-accelerated graph analytics including support for property graphs and scaling up to thousands of GPUs. cuGraph supports the creation and manipulation of graphs followed by the execution of scalable fast graph algorithms. It is part of the [RAPIDS](https://rapids.ai) accelerated data science framework.

[cuGraph GNN](https://github.com/rapidsai/cugraph-gnn) is a collection of GPU-accelerated plugins that support PyTorch and PyG natively through the _cuGraph-PyG_ and _WholeGraph_ subprojects. cuGraph GNN is built on top of cuGraph, leveraging its low-level [pylibcugraph](https://github.com/rapidsai/cugraph/python/pylibcugraph) API and C++ primitives for sampling and other GNN operations ([libcugraph](https://github.com/rapidai/cugraph/python/libcugraph)). It also includes the `libwholegraph` and `pylibwholegraph` libraries for high-performance distributed edgelist and embedding storage. Users have the option of working with these lower-level libraries directly, or through the higher-level API in cuGraph-PyG that directly implements the `GraphStore`, `FeatureStore`, `NodeLoader`, and `LinkLoader` interfaces.

Complete documentation on RAPIDS graph packages, including `cugraph`, `cugraph-pyg`, `pylibwholegraph`, and `pylibcugraph` is available on the [RAPIDS docs pages](https://docs.rapids.ai/api/cugraph/nightly/graph_support).

See [`rapidsai/cugraph-gnn/tree/branch-25.12/python/cugraph-pyg/cugraph_pyg/examples` on GitHub](https://github.com/rapidsai/cugraph-gnn/tree/branch-25.12/python/cugraph-pyg/cugraph_pyg/examples) for fully scalable PyG example workflows.
