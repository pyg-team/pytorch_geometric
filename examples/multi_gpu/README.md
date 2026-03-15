# Examples for Distributed Training

## Examples with NVIDIA GPUs

### Examples with cuGraph

For the best performance with NVIDIA GPUs, we recommend using **cuGraph**.
Refer to [our installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#accelerating-pyg-with-nvidia-cugraph-gnn) for setup instructions and to the [cuGraph-PyG examples](https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg/cugraph_pyg/examples) for ready-to-run training scripts covering single-node, multi-node, and link-prediction workloads.

### Examples with Pure PyTorch

| Example                                                                            | Scalability | Description                                                                                                                    |
| ---------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------ |
| [`distributed_batching.py`](./distributed_batching.py)                             | single-node | Training GNNs on multiple graphs.                                                                                              |
| [`distributed_sampling.py`](./distributed_sampling.py)                             | single-node | Training GNNs on a homogeneous graph with neighbor sampling.                                                                   |
| [`distributed_sampling_multinode.py`](./distributed_sampling_multinode.py)         | multi-node  | Training GNNs on a homogeneous graph with neighbor sampling on multiple nodes.                                                 |
| [`distributed_sampling_multinode.sbatch`](./distributed_sampling_multinode.sbatch) | multi-node  | Submitting a training job to a Slurm cluster using [`distributed_sampling_multinode.py`](./distributed_sampling_multinode.py). |
| [`papers100m_gcn.py`](./papers100m_gcn.py)                                         | single-node | Training GNNs on the `ogbn-papers100M` homogeneous graph w/ ~1.6B edges.                                                       |
| [`papers100m_gcn_multinode.py`](./papers100m_gcn_multinode.py)                     | multi-node  | Training GNNs on a homogeneous graph on multiple nodes.                                                                        |
| [`pcqm4m_ogb.py`](./pcqm4m_ogb.py)                                                 | single-node | Training GNNs for a graph-level regression task.                                                                               |
| [`mag240m_graphsage.py`](./mag240m_graphsage.py)                                   | single-node | Training GNNs on a large heterogeneous graph.                                                                                  |
| [`taobao.py`](./taobao.py)                                                         | single-node | Training link prediction GNNs on a heterogeneous graph.                                                                        |
| [`model_parallel.py`](./model_parallel.py)                                         | single-node | Model parallelism by manually placing layers on each GPU.                                                                      |

## Examples with Intel GPUs (XPUs)

| Example                                                        | Scalability            | Description                                                  |
| -------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------ |
| [`distributed_sampling_xpu.py`](./distributed_sampling_xpu.py) | single-node, multi-gpu | Training GNNs on a homogeneous graph with neighbor sampling. |
