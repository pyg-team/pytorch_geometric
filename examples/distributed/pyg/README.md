# Distributed Training with PyG

**[Distributed PyG](https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric/distributed)** implements scalable solution for distributed graph neural network (GNN) training, built exclusively on the powerful PyTorch Geometric (PyG) library. 

Current application can be deployed on a cluster of arbitrary size using multiple CPUs. For GPU-based solution refer to [GraphLearn-for-PyTorch (GLT)](https://github.com/pyg-team/pytorch_geometric/tree/5c0c2924a6c041db07d397547eff7fdf833a4ff8/examples/distributed/graphlearn_for_pytorch)

The solution is designed to effortlessly distribute the training of large-scale graph neural networks across multiple nodes, thanks to the integration of Distributed Data Parallelism (DDP) for model training and Remote Procedure Call (RPC) for efficient sampling and fetching of non-local features.
The design includes a number of custom classes, i.e. `DistNeighborSampler` that implements CPU sampling algorithms from local & remote data remaining consistent data structure at the output; an integrated `DistLoader` which ensures safe opening & closing of RPC connection between the samplers; a METIS-based `Partitioner` and many more.

## Distributed Sage for Ogbn-Product (Homo Multi-Node) Example

The example provided in `distributed_cpu.py` performs distributed training using [OGB](https://ogb.stanford.edu/) dataset and a `GraphSage` model. The example can run on both homogenous (ogbn-products) and heterogenous data (ogbn-mag). 
With minor modifications you can train on `ogbn-papers100M`, or any other dataset.

To run the example please refer to the steps below.

## Requirements

- Latest PyG and pyg-lib.
- Password-less SSH needs to be set up on all the nodes that you are using ([Linux SSH manual](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)).
- All nodes need to have a consistent environments installed, specifically torch and pyg-lib versions must be the same. You might want to consider using docker containers.
- [Optional] In some cases Linux firewall might be blocking TCP connection issues. Ensure that firewall settings allow for all nodes to communicate ([Linux firewall manual](https://ubuntu.com/server/docs/security-firewall)). For this example TCP ports 11111, 11112, 11113 should be open (i.e. `sudo ufw allow 11111`).


This example shows how to use distributed PyG to train PyG models in a distributed scenario with multiple CPU nodes. 

### Step 1: Prepare and partition the data

In distributed training, each node in the cluster holds a partition of the graph. Before the training starts, we partition the dataset into multiple partitions, each of which corresponds to a specific training node.

Here, we use `ogbn-products` and partition it into two partitions (in default) by the [partition example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/distributed/pyg/partition_graph.py) :

```bash
python partition_graph.py --dataset=ogbn-products --root_dir=./data/products --num_partitions=2
```
**Caution:** Partitioning with METIS is non-deterministic! 
All nodes should be able to access the same partition data, therefore generate partition on one node and copy the data to all members of the cluster, or place the folder in shared location.

The generated partition will have the folder below.

<img width="350" alt="partition-graph" src="https://github.com/pyg-team/pytorch_geometric/assets/58218729/2169e362-0259-4ac4-ab5e-8500b6b5bf4a">


### Step 2: Run the example in each training node

To run the example you can execute the commands in each node or use the provided launch script.

## Option A: Manual execution
You should change the `master_addr` to the IP of `node#0`. Make sure that correct `node_rank` is provided, with master node assigned rank 0. The `dataset_root_dir` should point to the head directory where your partition is placed, i.e. `../../data/partitions/ogbn-products/2-parts`.

```bash
# Node 0:
python distributed_cpu.py \
  --dataset=ogbn-products
  --dataset_root_dir=<partition folder directory> \
  --num_nodes=2 \
  --node_rank=0 \
  --master_addr=<master ip>

# Node 1:
python distributed_cpu.py \
  --dataset=ogbn-products
  --dataset_root_dir=<partition folder directory> \
  --num_nodes=2 \
  --node_rank=1 \
  --master_addr=<master ip>
```

## Option B: Launch script
TBD
See PR [#8241](https://github.com/pyg-team/pytorch_geometric/pull/8241)
