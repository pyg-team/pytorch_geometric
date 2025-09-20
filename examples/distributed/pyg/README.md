# Distributed Training with PyG

**[`torch_geometric.distributed`](https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric/distributed)** (deprecated) implements a scalable solution for distributed GNN training, built exclusively upon PyTorch and PyG.

Current application can be deployed on a cluster of arbitrary size using multiple CPUs.
PyG native GPU application is under development and will be released soon.

The solution is designed to effortlessly distribute the training of large-scale graph neural networks across multiple nodes, thanks to the integration of [Distributed Data Parallelism (DDP)](https://pytorch.org/docs/stable/notes/ddp.html) for model training and [Remote Procedure Call (RPC)](https://pytorch.org/docs/stable/rpc.html) for efficient sampling and fetching of non-local features.
The design includes a number of custom classes, *i.e.* (1) `DistNeighborSampler` implements CPU sampling algorithms and feature extraction from local and remote data remaining consistent data structure at the output, (2) an integrated `DistLoader` which ensures safe opening & closing of RPC connection between the samplers, and (3) a METIS-based `Partitioner` and many more.

## Example for Node-level Distributed Training on OGB Datasets

The example provided in [`node_ogb_cpu.py`](./node_ogb_cpu.py) performs distributed training with multiple CPU nodes using [OGB](https://ogb.stanford.edu/) datasets and a [`GraphSAGE`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html) model.
The example can run on both homogeneous (`ogbn-products`) and heterogeneous data (`ogbn-mag`).
With minor modifications, the example can be extended to train on `ogbn-papers100m` or any other dataset.

To run the example, please refer to the steps below.

### Requirements

- [`torch-geometric>=2.5.0`](https://github.com/pyg-team/pytorch_geometric) and [`pyg-lib>=0.4.0`](https://github.com/pyg-team/pyg-lib)
- Password-less SSH needs to be set up on all the nodes that you are using (see the [Linux SSH manual](https://linuxize.com/post/how-to-setup-passwordless-ssh-login)).
- All nodes need to have a consistent environments installed, specifically `torch` and `pyg-lib` versions must be the same.
  You might want to consider using docker containers.
- *[Optional]* In some cases Linux firewall might be blocking TCP connection issues.
  Ensure that firewall settings allow for all nodes to communicate (see the [Linux firewall manual](https://ubuntu.com/server/docs/security-firewall)).
  For this example TCP ports `11111`, `11112` and `11113` should be open (*i.e.* `sudo ufw allow 11111`).

### Step 1: Prepare and Partition the Data

In distributed training, each node in the cluster holds a partition of the graph.
Before the training starts, we partition the dataset into multiple partitions, each of which corresponds to a specific training node.

Here, we use `ogbn-products` and partition it into two partitions (in default) via the [`partition_graph.py`](./partition_graph.py) script:

```bash
python partition_graph.py --dataset=ogbn-products --root_dir=../../../data --num_partitions=2
```

**Caution:** Partitioning with METIS is non-deterministic!
All nodes should be able to access the same partition data.
Therefore, generate the partitions on one node and copy the data to all members of the cluster, or place the folder into a shared location.

The generated partition will have a folder structure as below:

```
data
├─ dataset
│  ├─ ogbn-mag
│  └─ ogbn-products
└─ partitions
   ├─ obgn-mag
   └─ obgn-products
      ├─ ogbn-products-partitions
      │  ├─ part_0
      │  ├─ part_1
      │  ├─ META.json
      │  ├─ node_map.pt
      │  └─ edge_map.pt
      ├─ ogbn-products-label
      │  └─ label.pt
      ├─ ogbn-products-test-partitions
      │  ├─ partition0.pt
      │  └─ partition1.pt
      └─ ogbn-products-train-partitions
         ├─ partition0.pt
         └─ partition1.pt
```

### Step 2: Run the Example in Each Training Node

To run the example, you can execute the commands in each node or use the provided launch script.

#### Option A: Manual Execution

You should change the `master_addr` to the IP of `node#0`.
Make sure that the correct `node_rank` is provided, with the master node assigned to rank `0`.
The `dataset_root_dir` should point to the head directory where your partition is placed, *i.e.* `../../data/partitions/ogbn-products/2-parts`:

```bash
# Node 0:
python node_ogb_cpu.py \
  --dataset=ogbn-products \
  --dataset_root_dir=<partition folder directory> \
  --num_nodes=2 \
  --node_rank=0 \
  --master_addr=<master ip>

# Node 1:
python node_obg_cpu.py \
  --dataset=ogbn-products \
  --dataset_root_dir=<partition folder directory> \
  --num_nodes=2 \
  --node_rank=1 \
  --master_addr=<master ip>
```

In some configurations, the network interface used for multi-node communication may be different than the default one.
In this case, the interface used for multi-node communication needs to be specified to Gloo.

Assuming that `$MASTER_ADDR` is set to the IP of `node#0`.

On the `node#0`:

```bash
export TP_SOCKET_IFNAME=$(ip addr | grep "$MASTER_ADDR" | awk '{print $NF}')
export GLOO_SOCKET_IFNAME=$TP_SOCKET_IFNAME
```

On the other nodes:

```bash
export TP_SOCKET_IFNAME=$(ip route get $MASTER_ADDR | grep -oP '(?<=dev )[^ ]+')
export GLOO_SOCKET_IFNAME=$TP_SOCKET_IFNAME
```

#### Option B: Launch Script

There exists two methods to run the distributed example with one script in one terminal for multiple nodes:

1. [`launch.py`](./launch.py):
   ```bash
   python launch.py
     --workspace {workspace}/pytorch_geometric
     --num_nodes 2
     --dataset_root_dir {dataset_dir}/mag/2-parts
     --dataset ogbn-mag
     --batch_size 1024
     --learning_rate 0.0004
     --part_config {dataset_dir}/mag/2-parts/ogbn-mag-partitions/META.json
     --ip_config {workspace}/pytorch_geometric/ip_config.yaml
    'cd /home/user_xxx; source {conda_envs}/bin/activate; cd {workspace}/pytorch_geometric; {conda_envs}/bin/python
     {workspace}/pytorch_geometric/examples/pyg/node_ogb_cpu.py --dataset=ogbn-mag --logging --progress_bar --ddp_port=11111'
   ```
1. [`run_dist.sh`](./run_dist.sh): All parameter settings are contained in the `run_dist.sh` script and you just need run with:
   ```bash
   ./run_dist.sh
   ```
