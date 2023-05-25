# Distributed (multi nodes) examples.

This example shows how to leverage [GraphLearn-for-PyTorch(GLT)](https://github.com/alibaba/graphlearn-for-pytorch) to train PyG models in distributed scenarios with GPUs. The dataset in this example is `ogbn-products` from the [Open Graph Benchmark](https://ogb.stanford.edu/) and you can also train on `ogbn-papers100M` with minor modifications of the configuration file.

To run this distributed example, you can choose to direct run the example or use our [script](launch.py) following the guidance below.

## Reference test environment and dependency packages
docker image: pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install --no-index  torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install graphlearn-torch
```

## Option 1: Directly run the example
### Step 1: Prepare and partition data.
Here we use ogbn_products and partition it into 2 partitions.
```
python partition_ogbn_dataset.py --dataset=ogbn-products --root_dir=../../../data/ogbn-products --num_partitions=2
```

### Step 2: Run the example of distributed training
2 nodes each with 2 GPUs
```
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_sage_supervised.py \
  --num_nodes=2 --node_rank=0 --master_addr=localhost \
  --dataset=ogbn-products --dataset_root_dir=../../../data/ogbn-products \
  --in_channel=100 --out_channel=47

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_sage_supervised.py \
  --num_nodes=2 --node_rank=1 --master_addr=localhost \
  --dataset=ogbn-products --dataset_root_dir=../../../data/ogbn-products \
  --in_channel=100 --out_channel=47
```

**Note:**
1. You should change master_addr to the ip of node#0
2. Since there is randomness in the step of partitioning, please ensure all nodes are using the same data when running `dist_train_sage_supervised.py`.


## Option 2: Use our script for the example
### Step 0: Setup a Distributed File System
**Note**: You may skip this step if you already set up folder(s) synchronized across machines.

To perform distributed sampling, files and codes need to be accessed across multiple machines. A distributed file system (i.e., NFS, Ceph, SSHFS, etc) exempt you from synchnonizing files such as partition information by hand.

For configuration details, please refer to the following links:

NFS: https://wiki.archlinux.org/index.php/NFS

SSHFS: https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh

Ceph: https://docs.ceph.com/en/latest/install/

### Step 1: Prepare and partition data
Here if we use ogbn-products and partition it into 2 partitions.
```
python partition_ogbn_dataset.py --dataset=ogbn-products --root_dir=../../../data/ogbn-products --num_partitions=2
```
And for ogbn-papers100M, we can set
```
python partition_ogbn_dataset.py --dataset=ogbn-papers100M --root_dir=../../../data/ogbn-papers100M --num_partitions=2
``` 
### Step 2: Set up the configure file
An example template for configure file is in the yaml format. e.g.
`dist_train_sage_sup_config.yml`.

```
# IP addresses for all nodes.
# Note: The first 3 params are expected to form usernames@nodes:ports
# to access each node by ssh
nodes:
  - 0.0.0.0
  - 1.1.1.1
# ssh ports for each node.
ports: [22, 22]
# username for remote IPs.
usernames:
  - your_username_for_node_0
  - your_username_for_node_1

# path to python with GLT envs for each node
python_bins:
  - /path/to/python
  - /path/to/python

# dataset name, e.g. ogbn-products, ogbn-papers100M.
# Note: make sure the name of dataset_root_dir the same as the dataset name.
dataset: ogbn-products

# in_channel and out_channel for the dataset
# The following two params are specific to datasets
# E.g.:
#   For ogbn-products: in_channel=100, out_channel=47
#   For ogbn-papers100M: in_channel=128, out_channel=172
in_channel: 100
out_channel: 47

# path to the pytorch_geometric directory
dst_paths:
  - /path/to/pytorch_geometric
  - /path/to/pytorch_geometric

# setup visible cuda devices for each node.
visible_devices:
  - 0,1,2,3
  - 0,1,2,3
```

### Step 3: Launch distributed jobs

```
pip install paramiko
pip install click
python launch.py --config=dist_train_sage_sup_config.yml --master_addr=0.0.0.0 --master_port=11234
```
Here `master_addr` is for the master address for RPC initialization, and `master_port` is for PyTorch's process group initialization across training processes.

Optional parameters which you can append after the command above includes:
```
--epochs: repeat epochs for sampling, (default=10)
--batch_size: batch size for sampling, (default=2048)
```

Note: you should change master_addr to the ip of server-node#0
