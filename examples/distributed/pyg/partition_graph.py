import argparse
import os
import os.path as osp

import torch
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, Reddit
from torch_geometric.distributed import Partitioner
from torch_geometric.utils import mask_to_index


def partition_dataset(
    dataset_name: str,
    root_dir: str,
    num_parts: int,
    recursive: bool = False,
    use_sparse_tensor: bool = False,
):
    abs_dir = '' if osp.isabs(root_dir) else osp.dirname(
        osp.realpath(__file__))
    data_dir = osp.join(abs_dir, root_dir)

    dataset_dir = osp.join(data_dir, 'dataset', dataset_name)
    dataset = get_dataset(dataset_name, dataset_dir, use_sparse_tensor)
    data = dataset[0]

    save_dir = osp.join(f'{data_dir}', 'partitions', f'{dataset_name}',
                        f'{num_parts}-parts')

    partitions_dir = osp.join(save_dir, f'{dataset_name}-partitions')
    partitioner = Partitioner(data, num_parts, partitions_dir, recursive)
    partitioner.generate_partition()

    print('-- Saving label ...')
    label_dir = osp.join(save_dir, f'{dataset_name}-label')
    os.makedirs(label_dir, exist_ok=True)

    split_data = data['paper'] if dataset_name == 'ogbn-mag' else data

    torch.save(split_data.y.squeeze(), osp.join(label_dir, 'label.pt'))

    train_idx, valid_idx, test_idx = get_idx_split(dataset, dataset_name,
                                                   split_data)

    print('-- Partitioning training indices ...')
    train_idx = train_idx.split(train_idx.size(0) // num_parts)
    train_part_dir = osp.join(save_dir, f'{dataset_name}-train-partitions')
    os.makedirs(train_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(train_idx[i], osp.join(train_part_dir, f'partition{i}.pt'))

    print('-- Partitioning validation indices ...')
    valid_idx = valid_idx.split(valid_idx.size(0) // num_parts)
    valid_part_dir = osp.join(save_dir, f'{dataset_name}-valid-partitions')
    os.makedirs(valid_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(valid_idx[i], osp.join(valid_part_dir, f'partition{i}.pt'))

    print('-- Partitioning test indices ...')
    test_idx = test_idx.split(test_idx.size(0) // num_parts)
    test_part_dir = osp.join(save_dir, f'{dataset_name}-test-partitions')
    os.makedirs(test_part_dir, exist_ok=True)
    for i in range(num_parts):
        torch.save(test_idx[i], osp.join(test_part_dir, f'partition{i}.pt'))


def get_dataset(name, dataset_dir, use_sparse_tensor=False):
    transform = T.ToSparseTensor(
        remove_edge_index=False) if use_sparse_tensor else None
    if name == 'ogbn-mag':
        if transform is None:
            transform = T.ToUndirected(merge=True)
        else:
            transform = T.Compose([T.ToUndirected(merge=True), transform])
        dataset = OGB_MAG(root=dataset_dir, preprocess='metapath2vec',
                          transform=transform)
    elif name == 'ogbn-products':
        if transform is None:
            transform = T.RemoveDuplicatedEdges()
        else:
            transform = T.Compose([T.RemoveDuplicatedEdges(), transform])

        dataset = PygNodePropPredDataset('ogbn-products', root=dataset_dir,
                                         transform=transform)

    elif name == 'Reddit':
        dataset = Reddit(root=dataset_dir, transform=transform)

    return dataset


def get_idx_split(dataset, dataset_name, split_data):
    if dataset_name == 'ogbn-mag' or dataset_name == 'Reddit':
        train_idx = mask_to_index(split_data.train_mask)
        test_idx = mask_to_index(split_data.test_mask)
        valid_idx = mask_to_index(split_data.val_mask)

    elif dataset_name == 'ogbn-products':
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        test_idx = split_idx['test']
        valid_idx = split_idx['valid']

    return train_idx, valid_idx, test_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--dataset', type=str, choices=['ogbn-mag', 'ogbn-products', 'Reddit'],
        default='ogbn-products')
    add('--root_dir', default='../../../data', type=str,
        help='relative path to look for the datasets')
    add('--num_partitions', type=int, default=4)
    add('--recursive', action='store_true')
    # TODO (kgajdamo) add support for arguments below.
    # add('--use-sparse-tensor', action='store_true',
    #     help='use torch_sparse.SparseTensor as graph storage format')
    # add('--bf16', action='store_true')
    args = parser.parse_args()

    partition_dataset(args.dataset, args.root_dir, args.num_partitions,
                      args.recursive)
