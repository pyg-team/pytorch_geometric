# Using GraphLearn-for-PyTorch (GLT) for Distributed Training with PyG

**[GraphLearn-for-PyTorch (GLT)](https://github.com/alibaba/graphlearn-for-pytorch)** is a graph learning library for PyTorch that makes distributed GNN training easy and efficient.
GLT leverages GPUs to accelerate graph sampling and utilizes UVA and GPU caches to reduce the data conversion and transferring costs during graph sampling and model training.
Most of the APIs of GLT are compatible with PyG, so PyG users only need to modify a few lines of their PyG code to train their model with GLT.

## Requirements

- `python >= 3.6`
- `torch >= 1.12`
- `graphlearn-torch`

## Distributed (Multi-Node) Example

This example shows how to leverage [GraphLearn-for-PyTorch (GLT)](https://github.com/alibaba/graphlearn-for-pytorch) to train PyG models in a distributed scenario with GPUs. The dataset in this example is `ogbn-products` from the [Open Graph Benchmark](https://ogb.stanford.edu/), but you can also train on `ogbn-papers100M` with only minor modifications.

To run this example, you can run the example as described below or directly make use of our [`launch.py`](launch.py) script.
The training results will be generated and saved in `dist_sage_sup.txt`.

### Running the Example

#### Step 1: Prepare and partition the data

Here, we use `ogbn-products` and partition it into two partitions:

```bash
python partition_ogbn_dataset.py --dataset=ogbn-products --root_dir=../../../data/ogbn-products --num_partitions=2
```

#### Step 2: Run the example in each training node

For example, running the example in two nodes each with two GPUs:

```bash
# Node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_sage_supervised.py \
  --num_nodes=2 --node_rank=0 --master_addr=localhost \
  --dataset=ogbn-products --dataset_root_dir=../../../data/ogbn-products \
  --in_channel=100 --out_channel=47

# Node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_sage_supervised.py \
  --num_nodes=2 --node_rank=1 --master_addr=localhost \
  --dataset=ogbn-products --dataset_root_dir=../../../data/ogbn-products \
  --in_channel=100 --out_channel=47
```

**Notes:**

1. You should change the `master_addr` to the IP of `node#0`.
2. Since there is randomness during data partitioning, please ensure all nodes are using the same partitioned data when running `dist_train_sage_supervised.py`.

### Using the `launch.py` Script

#### Step 1: Setup a distributed file system

**Note**: You may skip this step if you already set up folder(s) synchronized across machines.

To perform distributed sampling, files and codes need to be accessed across multiple machines.
A distributed file system (*i.e.*, [NFS](https://wiki.archlinux.org/index.php/NFS), [SSHFS](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh), [Ceph](https://docs.ceph.com/en/latest/install), ...) exempts you from synchnonizing files such as partition information.

#### Step 2: Prepare and partition the data

In distributed training (under the worker mode), each node in the cluster holds a partition of the graph.
Thus, before the training starts, we partition the `ogbn-products` dataset into multiple partitions, each of which corresponds to a specific training worker.

The partitioning occurs in three steps:
  1. Run the partition algorithm to assign nodes to partitions.
  2. Construct the partitioned graph structure based on the node assignment.
  3. Split the node features and edge features into partitions.

GLT supports caching graph topology and frequently accessed features in GPU to accelerate GPU sampling and feature collection.
For feature caching, we adopt a pre-sampling-based approach to determine the hotness of nodes, and cache features for nodes with higher hotness while loading the graph.
The uncached features are stored in pinned memory for efficient access via UVA.

For further information about partitioning, please refer to the [official tutorial](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/docs/tutorial/dist.md).

Here, we use `ogbn-products` and partition it into two partitions:

```bash
python partition_ogbn_dataset.py --dataset=ogbn-products --root_dir=../../../data/ogbn-products --num_partitions=2
```

#### Step 3: Set up the configure file

An example configuration file in given via [`dist_train_sage_sup_config.yml`](dist_train_sage_sup_config.yml).

#### Step 4: Launch the distributed training

```bash
pip install paramiko
pip install click
apt install tmux
python launch.py --config=dist_train_sage_sup_config.yml --master_addr=0.0.0.0 --master_port=11234
```

Here, `master_addr` is for the master RPC address, and `master_port` is for PyTorch's process group initialization across training processes.
Note that you should change the `master_addr` to the IP of `node#0`.
