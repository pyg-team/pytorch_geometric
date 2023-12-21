import argparse
import os
import os.path as osp

import torch

from benchmark.utils import get_dataset_with_transformation
from torch_geometric.distributed import Partitioner
from torch_geometric.utils import mask_to_index


def partition_dataset(
    dataset_name: str,
    root_dir: str,
    num_parts: int,
    recursive: bool = False,
    use_sparse_tensor: bool = False,
    bf16: bool = False,
):
    abs_dir = '' if osp.isabs(root_dir) else osp.dirname(
        osp.realpath(__file__))
    data_dir = osp.join(abs_dir, root_dir)

    dataset_dir = osp.join(data_dir, 'dataset')

    data = get_dataset_with_transformation(dataset_name, dataset_dir,
                                           use_sparse_tensor, bf16)[0]

    save_dir = osp.join(f'{data_dir}', 'partitions', f'{dataset_name}',
                        f'{num_parts}-parts')

    partitions_dir = osp.join(save_dir, f'{dataset_name}-partitions')
    partitioner = Partitioner(data, num_parts, partitions_dir, recursive)
    partitioner.generate_partition()

    print('-- Saving label ...')
    label_dir = osp.join(save_dir, f'{dataset_name}-label')
    os.makedirs(label_dir, exist_ok=True)

    save_data = data['paper'] if dataset_name == 'ogbn-mag' else data

    torch.save(save_data.y.squeeze(), osp.join(label_dir, 'label.pt'))

    print('-- Partitioning training indices ...')
    train_idx = mask_to_index(save_data.train_mask)
    train_idx = train_idx.split(train_idx.size(0) // num_parts)
    train_part_dir = osp.join(save_dir, f'{dataset_name}-train-partitions')
    os.makedirs(train_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(train_idx[i], osp.join(train_part_dir, f'partition{i}.pt'))

    print('-- Partitioning test indices ...')
    test_idx = mask_to_index(save_data.test_mask)
    test_idx = test_idx.split(test_idx.size(0) // num_parts)
    test_part_dir = osp.join(save_dir, f'{dataset_name}-test-partitions')
    os.makedirs(test_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(test_idx[i], osp.join(test_part_dir, f'partition{i}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--dataset', type=str, default='ogbn-products')
    add('--root_dir', default='../../../data', type=str,
        help='relative path to look for the datasets')
    add('--num_partitions', type=int, default=2)
    add('--recursive', action='store_true')
    # TODO (kgajdamo) add support for the below arguments.
    # add('--use-sparse-tensor', action='store_true',
    #     help='use torch_sparse.SparseTensor as graph storage format')
    # add('--bf16', action='store_true')
    args = parser.parse_args()

    partition_dataset(args.dataset, args.root_dir, args.num_partitions,
                      args.recursive)
