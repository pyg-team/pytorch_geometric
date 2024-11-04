# Distributed Training with PyG using NVIDIA RAPIDS libraries

This directory contains examples for distributed graph learning using NVIDIA RAPIDS cuGraph/WholeGraph libraries. These examples minimize CPU interruptions and maximize GPU throughput advantages during the graph dataloading stage. In our tests, we normally observe at least over a tenfold speedup compared to traditional CPU-based [RPC methods](../pyg). Additionally, the libraries are user-friendly, enabling flexible integration with minimal effort to upgrade from users' GNN training workflows.

Currently, we offer two integration options for NVIDIA RAPIDS support: the first is through cuGraph's high-performance, compact dataloader API, and the second approach is through WholeGraph, which accelerates feature and graph store accesses independently by leveraging PyG remote backend APIs, showcasing the integration with user-customized GNN workflows or GraphML tasks. We plan to merge these two paths soon under [cugraph-gnn](https://github.com/rapidsai/cugraph-gnn), to simplify the user learning curve.

1. [`cuGraph`](./cugraph): Distributed training via NVIDIA RAPIDS [cuGraph](https://github.com/rapidsai/cugraph) library.
1. [`WholeGraph`](./wholegraph): Distributed training via PyG remote backend APIs and NVIDIA RAPIDS [WholeGraph](https://github.com/rapidsai/wholegraph) library.
