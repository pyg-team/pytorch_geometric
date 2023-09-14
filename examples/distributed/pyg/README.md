# Distributed Training with PyG

**[Distributed PyG](https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric/distributed)** is one distributed solution for torch_geometric on multi-nodes which can significantly reduce the training time with the increasing of node numbers. The solution provides the whole training steps including partition dataset preparation, environment setup, running the training for distributed, etc. There are several examples to show how to run the distributed GNN training by distributed PyG including distributed sage for ogbn-product (homo), distributed sage for ogbn-mags (heter), edge sampling, etc.

## Requirements

- latest PyG

## Distributed Sage for Ogbn-Product (Homo Multi-Node) Example

This example shows how to use [distributed PyG](https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric/distributed) to train PyG models in a distributed scenario with multiple CPU nodes. The dataset in this example is `ogbn-products` from the [Open Graph Benchmark](https://ogb.stanford.edu/), but you can also train on `ogbn-papers100M` with only minor modifications.

### Step 1: Setup a distributed environment

#### Requirements

1) Password-less ssh needs to be set up on all the nodes that you are using.
2) A network file system (NFS) is set up for all the nodes to access.

To perform distributed sampling, files and codes need to be accessed across multiple machines.
A distributed file system (*i.e.*, [NFS](https://wiki.archlinux.org/index.php/NFS), [SSHFS](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh), [Ceph](https://docs.ceph.com/en/latest/install), ...) is required to allow you for synchnonizing files such as partition information.


### Step 2: Prepare and partition the data

In distributed training, each node in the cluster holds a partition of the graph. Before the training starts, we partition the `ogbn-products` dataset into multiple partitions, each of which corresponds to a specific training node.

Here, we use `ogbn-products` and partition it into two partitions (in default) by the [partition example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/distributed/pyg/partition_graph.py) :

```bash
python partition_graph.py --dataset=ogbn-products --root_dir=./data/products --num_partitions=2
```
The generated partition will have the folder below.

![image](https://github.com/ZhengHongming888/pytorch_geometric/assets/33777424/1d375496-99d5-43f9-b1c3-ce345e2572c0)


You can put/move the products partition folder into one public folder that each node can access this shared folder.




### Step 3: Run the example in each training node

For example, running the example in two nodes each with two GPUs:

```bash
# Node 0:
python dist_train_sage_for_homo.py \
  --dataset_root_dir=your partition folder \
  --num_nodes=2 --node_rank=0 --num_training_procs=1 \
  --master_addr= master ip

# Node 1:
python dist_train_sage_for_homo.py \
  --dataset_root_dir=your partition folder \
  --num_nodes=2 --node_rank=1 --num_training_procs=1 \
  --master_addr= master ip
```

**Notes:**

1. You should change the `master_addr` to the IP of `node#0`.
2. In default this example will use the num_workers = 2 for number of sampling workers and concurrency=2 for mp.queue. you can also add these argument to speed up the training like "--num_workers=8 --concurrency=8"
3. All nodes need to use the same partitioned data when running `dist_train_sage_for_homo.py`.




## Distributed Sage for Ogbn-Mags (Hetero Multi-Node) Example

### Step 1: Setup a distributed environment

Same as Homo above.

### Step 2: Prepare and partition the data


Here, we use `ogbn-products` and partition it into two partitions (in default) by the [partition example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/distributed/pyg/partition_hetero_graph.py) :

```bash
python partition_hetero_graph.py --dataset=ogbn-mag --root_dir=./data/mag --num_partitions=2
```
The generated partition will have the folder below.

![image](https://github.com/ZhengHongming888/pytorch_geometric/assets/33777424/4d2e3212-d3e4-4f5a-bc91-38e92e32422e)


You can put/move the products partition folder into one public folder that each node can access this shared folder.


### Step 3: Run the example in each training node

For example, running the example in two nodes each with two GPUs:

```bash
# Node 0:
python dist_train_sage_for_hetero.py \
  --dataset_root_dir=your partition folder \
  --dataset=ogbn-mags \
  --num_nodes=2 --node_rank=0 --num_training_procs=1 \
  --master_addr= master ip

# Node 1:
python dist_train_sage_for_hetero.py \
  --dataset_root_dir=your partition folder \
  --dataset=ogbn-mags \
  --num_nodes=2 --node_rank=1 --num_training_procs=1 \
  --master_addr= master ip
```
