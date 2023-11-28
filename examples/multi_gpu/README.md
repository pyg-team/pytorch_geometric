# Examples for Distributed Training

## Examples with Nvidia GPUs

| Example                               | Scalability | Description                                                                                                                                                       |
| ------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| distributed_batching.py               | single-node | Example for training GNNs on multiple graphs.                                                                                                                     |
| distributed_sampling.py               | single-node | Example for training GNNs on a homogeneous graph with neighbor sampling.                                                                                          |
| distributed_sampling_multinode.py     | multi-node  | Example for training GNNs on a homogeneous graph with neighbor sampling on multiple nodes.                                                                        |
| distributed_sampling_multinode.sbatch | multi-node  | Example for submitting a training job to a Slurm cluster using `distributed_sampling_multi_node.py`.                                                              |
| papers100m_gcn.py                     | single-node | Example for training GNNs on a homogeneuos graph.                                                                                                                 |
| papers100m_gcn_multinode.py           | multi-node  | Example for training GNNs on a homogeneous graph on multiple nodes.                                                                                               |
| mag240m_graphsage.py                  | single-node | Example for training GNNs on a heterogeneous graph.                                                                                                               |
| taobao.py                             | single-node | Example for training GNNs on a heterogeneous graph.                                                                                                               |
| model_parallel.py                     | single-node | Example for model parallelism by manually placing layers on each GPU.                                                                                             |
| data_parallel.py                      | single-node | `torch.nn.DataParallel` is discouraged ([pytorch/pytorch#65936](https://github.com/pytorch/pytorch/issues/65936)). |

## Examples with Intel GPUs (XPUs)

| Example                     | Scalability            | Description                                                              |
| --------------------------- | ---------------------- | ------------------------------------------------------------------------ |
| distributed_sampling_xpu.py | single-node, multi-gpu | Example for training GNNs on a homogeneous graph with neighbor sampling. |
