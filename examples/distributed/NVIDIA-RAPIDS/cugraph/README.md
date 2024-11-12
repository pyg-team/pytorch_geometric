# Distributed Training with cuGraph-PyG

[RAPIDS cuGraph-PyG](https://github.com/rapidsai/cugraph-gnn) is an extension for PyG that supports accelerated data loading and feature fetching. It is powered by [RAPIDS cuGraph](https://github.com/rapidsai/cugraph)

cuGraph-PyG supports single-GPU, multi-GPU, and multi-node/multi-GPU training and sampling using native PyG APIs.

## Requirements

- **PyTorch**: `>= 2.3`
- **TensorDict**: `>= 0.1.2`
- **cuGraph-PyG**: `>= 24.08`
- **CUDA**: `>= 11.8`
- **NVIDIA GPU(s) with compute capability 7.0+**

For more documentation on how to install RAPIDS, see this [guide](https://docs.rapids.ai/install/).

## How to Run

To run, use sbatch
(i.e. `sbatch -N2 -p <partition> -A <account> -J <job name>`)
with the script shown below:

```
 head_node_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

 (yes || true) | srun -l \
        --container-image <container image> \
        --container-mounts "$(pwd):/workspace","/raid:/raid" \
          torchrun \
          --nnodes 2 \
          --nproc-per-node 8 \
          --rdzv-backend c10d \
          --rdzv-id 62 \
          --rdzv-endpoint $head_node_addr:29505 \
          /workspace/papers100m_gcn_cugraph_multinode.py \
            --epochs 1 \
            --dataset ogbn-papers100M \
            --dataset_root /workspace/datasets \
            --tempdir_root /raid/scratch
```
