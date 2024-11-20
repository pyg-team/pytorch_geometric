# Using NVIDIA WholeGraph Library for Distributed Training with PyG

**[RAPIDS WholeGraph](https://github.com/rapidsai/wholegraph)**
NVIDIA WholeGraph is designed to optimize the training of Graph Neural Networks (GNNs) that are often constrained by data loading operations. It provides an underlying storage structure, called WholeMemory, which efficiently manages data storage/communication across disk, RAM, and device memory by leveraging NVIDIA GPUs and communication libraries like NCCL/NVSHMEM.

WholeGraph is a low-level graph storage library, integrated into and able to work alongside cuGraph, that directly provides an efficient feature and graph store with associated primitive operations (e.g., GPU-accelerated fast embedding retrieval and graph sampling). It is specifically optimized for NVLink systems, including DGX, MGX, and GH/GB200 machine or clusters.

This example demonstrates how to use WholeGraph to easily distribute the graph and feature store to pinned-host memory for fast GPU UVA access (see the DistTensor class), eliminating the need for manual graph partitioning or any custom third-party launch scripts. WholeGraph seamlessly integrates with PyTorch's Distributed Data Parallel (DDP) setup and works with standard distributed job launchers such as torchrun, mpirun, or srun.

## Requirements

- **PyTorch**: `>= 2.0`
- **PyTorch Geometric**: `>= 2.0.0`
- **WholeGraph**: `>= 24.02`
- **NVIDIA GPU(s)**

## Environment Setup

```bash
  pip install pylibwholegraph-cu12
```

## Dataset Preparation

For large dataset like ogbn-papers100M, downloading and preprocessing the dataset may take very long time. Users can run the following command to download and preprocess the dataset in advance:

```bash
python download_papers.py
```

If data is not pre-downloaded this may cause issues.

## Sinlge/Multi-GPU Run

Using PyTorch torchrun elastic launcher:

```
torchrun papers100m_dist_wholegraph_nc.py
```

or, using multi-GPUs if applicable:

```
torchrun --nnodes 1 --nproc-per-node <ngpu_per_node> papers100m_dist_wholegraph_nc.py
```

## Distributed (multi-node) Run

For example, let's use the slurm launcher here:

```
srun -N<num_nodes> --ntasks-per-node=<ngpu_per_node> python papers100m_dist_wholegraph_nc.py
```

Note the above command line setting is simplified for demonstration purposes. For more details, please refer to this [sbatch script](https://github.com/chang-l/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling_multinode.sbatch), as cluster setups may vary.

## Benchmark Run

The benchmark script is similar to the above example but includes a `--mode` command-line argument, allowing users to easily compare PyG's native features/graph store (`torch_geometric.data.Data` and `torch_geometric.data.HeteroData`) with the WholeMemory-based feature store and graph store, shown in this example. It performs a node classification task on the `ogbn-products` dataset.

### PyG baseline

```
torchrun --nnodes 1 --nproc-per-node <ngpu_per_node> benchmark_data.py --mode baseline
```

### WholeGraph FeatureStore integration (UVA for feature store access)

```
torchrun --nnodes 1 --nproc-per-node <ngpu_per_node> benchmark_data.py --mode UVA-features
```

### WholeGraph FeatureStore + GraphStore (UVA for feature and graph store access)

```
torchrun --nnodes 1 --nproc-per-node <ngpu_per_node> benchmark_data.py  --mode UVA
```