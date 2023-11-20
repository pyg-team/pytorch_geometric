import argparse
import os
import os.path as osp

import torch

from torch_geometric.datasets import OGB_MAG
from torch_geometric.distributed import Partitioner


def partition_dataset(
    ogbn_dataset: str,
    root_dir: str,
    num_parts: int,
    recursive: bool = False,
):
    save_dir = osp.join(root_dir, f'{ogbn_dataset}-partitions')
    dataset = OGB_MAG(root=ogbn_dataset, preprocess='metapath2vec')
    data = dataset[0]

    partitioner = Partitioner(data, num_parts, save_dir, recursive)
    partitioner.generate_partition()

    print('-- Saving label ...')
    label_dir = osp.join(root_dir, f'{ogbn_dataset}-label')
    os.makedirs(label_dir, exist_ok=True)
    torch.save(data['paper'].y.squeeze(), osp.join(label_dir, 'label.pt'))

    print('-- Partitioning training indices ...')
    train_idx = data['paper'].train_mask.nonzero().view(-1)
    train_idx = train_idx.split(train_idx.size(0) // num_parts)
    train_part_dir = osp.join(root_dir, f'{ogbn_dataset}-train-partitions')
    os.makedirs(train_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(train_idx[i], osp.join(train_part_dir, f'partition{i}.pt'))

    print('-- Partitioning test indices ...')
    test_idx = data['paper'].test_mask.nonzero().view(-1)
    test_idx = test_idx.split(test_idx.size(0) // num_parts)
    test_part_dir = osp.join(root_dir, f'{ogbn_dataset}-test-partitions')
    os.makedirs(test_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(test_idx[i], osp.join(test_part_dir, f'partition{i}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-mag')
    parser.add_argument('--root_dir', type=str, default='./data/mag')
    parser.add_argument('--num_partitions', type=int, default=2)
    parser.add_argument('--recursive', type=bool, default=False)
    args = parser.parse_args()

    partition_dataset(args.dataset, args.root_dir, args.num_partitions,
                      args.recursive)
