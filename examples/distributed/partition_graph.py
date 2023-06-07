import argparse
import os.path as osp

import torch
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.distributed.partition.partitioner import (
    Partitioner,
    prepare_directory,
)


def partition_dataset(ogbn_dataset: str, root_dir: str, num_partitions: int):
    save_dir = root_dir + "/partition"
    dataset = PygNodePropPredDataset(ogbn_dataset, root_dir)
    data = dataset[0]

    partitioner = Partitioner(output_dir=save_dir, num_parts=num_partitions,
                              data=data)
    partitioner.generate_partition()
    split_idx = dataset.get_idx_split()
    print('-- Saving label ...')
    label_dir = prepare_directory(root_dir, f'{ogbn_dataset}-label')
    torch.save(data.y.squeeze(), osp.join(label_dir, 'label.pt'))

    print('-- Partitioning training idx ...')
    train_idx = split_idx['train']
    train_idx = train_idx.split(train_idx.size(0) // num_partitions)
    train_idx_partitions_dir = prepare_directory(
        root_dir, f'{ogbn_dataset}-train-partitions')
    for pidx in range(num_partitions):
        torch.save(train_idx[pidx],
                   osp.join(train_idx_partitions_dir, f'partition{pidx}.pt'))

    print('-- Partitioning test idx ...')
    test_idx = split_idx['test']
    test_idx = test_idx.split(test_idx.size(0) // num_partitions)
    test_idx_partitions_dir = prepare_directory(
        root_dir, f'{ogbn_dataset}-test-partitions')
    for pidx in range(num_partitions):
        torch.save(test_idx[pidx],
                   osp.join(test_idx_partitions_dir, f'partition{pidx}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Arguments for ClusterData Partitioning.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='ogbn-products',
        help="The name of dataset.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default='./data/products',
        help="The root directory of input dataset and output partitions.",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=2,
        help="Number of partitions",
    )
    args = parser.parse_args()

    partition_dataset(
        ogbn_dataset=args.dataset,
        root_dir=osp.join(osp.dirname(osp.realpath(__file__)),
                          args.root_dir), num_partitions=args.num_partitions)
