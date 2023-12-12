from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.distributed import (
    Partitioner,
)
import torch_geometric.transforms as T
import torch
import os
import os.path as osp

dataset_name = 'fake_hetero'
num_parts = 2

root_dir = f"/home/pyg/graphlearn-dev/data/{dataset_name}"

data = FakeHeteroDataset(
    num_graphs=1,
    avg_num_nodes=1000,
    avg_degree=3,
    num_node_types=2,
    num_edge_types=4,
    edge_dim=2,
)[0]

partitioner = Partitioner(data, num_parts, root_dir)
partitioner.generate_partition()

print('-- Saving label ...')
label_dir = osp.join(root_dir, f'{dataset_name}-label')
os.makedirs(label_dir, exist_ok=True)
torch.save(data['v0'].y.squeeze(), osp.join(label_dir, 'label.pt'))

print('-- Partitioning training indices ...')
train_idx = data['v0'].train_mask.nonzero().view(-1)
train_idx = train_idx.split(train_idx.size(0) // num_parts)
train_part_dir = osp.join(root_dir, f'{dataset_name}-train-partitions')
os.makedirs(train_part_dir, exist_ok=True)
for i in range(num_parts):
    torch.save(train_idx[i], osp.join(train_part_dir, f'partition{i}.pt'))

print('-- Partitioning test indices ...')
test_idx = data['v0'].test_mask.nonzero().view(-1)
test_idx = test_idx.split(test_idx.size(0) // num_parts)
test_part_dir = osp.join(root_dir, f'{dataset_name}-test-partitions')
os.makedirs(test_part_dir, exist_ok=True)
for i in range(num_parts):
    torch.save(test_idx[i], osp.join(test_part_dir, f'partition{i}.pt'))

# data_path = "/home/pyg/graphlearn-dev/partition_fake_und"

# data = FakeHeteroDataset(
#     num_graphs=1,
#     avg_num_nodes=100,
#     avg_degree=3,
#     num_node_types=2,
#     num_edge_types=4,
#     edge_dim=2,
#     transform=T.ToUndirected())[0]

# num_parts = 2
# partitioner = Partitioner(data, num_parts, data_path)
# partitioner.generate_partition()
