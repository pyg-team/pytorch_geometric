# Distributed (multi nodes) examples.

This example shows how to leverage [GraphLearn-for-PyTorch(GLT)](https://github.com/alibaba/graphlearn-for-pytorch) to train PyG models in distributed scenarios. The dataset in this example is `ogbn-products` from the [Open Graph Benchmark](https://ogb.stanford.edu/) and you can also train on `ogbn-papers100M` with minor modifications of the configuration file.

To run this distributed example, you can choose to direct run the example or use our [script](run_dist_train_sage_sup.py) following the guidance below.

## Reference test environment and dependency packages
docker image: pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install --no-index  torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install graphlearn-torch
```

## Option 1: Directly run the example
### Step 1: Prepare data.
Here we use ogbn_products and partition it into 2 partitions.
```
python partition_ogbn_dataset.py --dataset=ogbn-products --num_partitions=2
```

### Example of distributed training
2 nodes each with 2 GPUs
```
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_sage_supervised.py \
  --num_nodes=2 --node_rank=0 --master_addr=localhost

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_sage_supervised.py \
  --num_nodes=2 --node_rank=1 --master_addr=localhost
```

**Note:** you should change master_addr to the ip of node#0 and ensure the consistency of data between nodes when running the example.

## Option 2: Use our script for the example
### Step 0: Setup a Distributed File System
**Note**: You may skip this step if you already set up folder(s) synchronized across machines.

To perform distributed sampling, files and codes need to be accessed across multiple machines. A distributed file system (i.e., NFS, Ceph, SSHFS, etc) exempt you from synchnonizing files such as partition information by hand.

For configuration details, please refer to the following links:

NFS: https://wiki.archlinux.org/index.php/NFS

SSHFS: https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh

Ceph: https://docs.ceph.com/en/latest/install/

### Step 1: Prepare data
Here we use ogbn_products and partition it into 2 partitions.
```
python partition_ogbn_dataset.py --dataset=ogbn-products --num_partitions=2
```

### Step 2: Set up the configure file
An example template for configure file is in the yaml format. e.g.
`run_dist_train_sage_sup.yml`.

```
# IP addresses for all nodes.
nodes:
  - 0.0.0.0
  - 1.1.1.1

# ssh IP ports for each node.
ports: [22, 22]

# path to python with GLT envs
python_bins:
  - /path/to/python
  - /path/to/python
# username for remote IPs.
usernames:
  - root
  - root

# dataset name, e.g. ogbn-products.
# Note: make sure the name of dataset_root_dir the same as the dataset name.
dataset: ogbn-products

# in_channel and out_channel for the dataset
in_channel: 100
out_channel: 47

node_ranks: [0, 1]

dst_paths:
  - /path/to/GLT
  - /path/to/GLT

# Setup visible cuda devices for each node.
visible_devices:
  - 0,1,2,3
  - 4,5,6,7
```

### Step 3: Launch distributed jobs

```
pip install paramiko
pip install click
python run_dist_train_sage_sup.py --config=dist_train_sage_sup_config.yml --master_addr=0.0.0.0 --master_port=11234
```

Optional parameters which you can append after the command above includes:
```
--epochs: repeat epochs for sampling, (default=1)
--batch_size: batch size for sampling, (default=2048)

```


Here we present a comparison of end-to-end training time, as well as
sampling and feature lookup time between GLT and DGL.
We conducted the comparison using DGL version 0.9.1 and GLT version 0.2.0rc2
in an environment consisting of 2 nodes, each with 2 A100 GPUs.

<p align="center">
  <img width="60%" src=../../docs/figures/glt_dgl.png />
</p>



## Example of distributed training with server-client mode
- 2 server nodes each with 1 server process for remote sampling and 2 GPUs.
- 2 client nodes each with 2 client processes for training and 2 GPUs.
```
# server node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_sage_supervised_with_server.py \
  --num_server_nodes=2 --num_client_nodes=2 --role=server --node_rank=0 \
  --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost

# server node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_sage_supervised_with_server.py \
  --num_server_nodes=2 --num_client_nodes=2 --role=server --node_rank=1 \
  --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost

# client node 0:
CUDA_VISIBLE_DEVICES=4,5 python dist_train_sage_supervised_with_server.py \
  --num_server_nodes=2 --num_client_nodes=2 --role=client --node_rank=0 \
  --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost

# client node 1:
CUDA_VISIBLE_DEVICES=6,7 python dist_train_sage_supervised_with_server.py \
  --num_server_nodes=2 --num_client_nodes=2 --role=client --node_rank=1 \
  --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost
```

Note: you should change master_addr to the ip of server-node#0