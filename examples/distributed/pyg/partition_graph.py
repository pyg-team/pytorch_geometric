import argparse
import os
import os.path as osp

import torch
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, MovieLens, Reddit
from torch_geometric.distributed import Partitioner
from torch_geometric.utils import mask_to_index


def partition_dataset(
    dataset_name: str,
    root_dir: str,
    num_parts: int,
    recursive: bool = False,
    use_sparse_tensor: bool = False,
):
    if not osp.isabs(root_dir):
        path = osp.dirname(osp.realpath(__file__))
        root_dir = osp.join(path, root_dir)

    dataset_dir = osp.join(root_dir, 'dataset', dataset_name)
    dataset = get_dataset(dataset_name, dataset_dir, use_sparse_tensor)
    data = dataset[0]

    save_dir = osp.join(root_dir, 'partitions', dataset_name,
                        f'{num_parts}-parts')

    partitions_dir = osp.join(save_dir, f'{dataset_name}-partitions')
    partitioner = Partitioner(data, num_parts, partitions_dir, recursive)
    partitioner.generate_partition()

    print('-- Saving label ...')
    label_dir = osp.join(save_dir, f'{dataset_name}-label')
    os.makedirs(label_dir, exist_ok=True)

    if dataset_name == 'ogbn-mag':
        split_data = data['paper']
        label = split_data.y
    else:
        split_data = data
        if dataset_name == 'ogbn-products':
            label = split_data.y.squeeze()
        elif dataset_name == 'Reddit':
            label = split_data.y
        elif dataset_name == 'MovieLens':
            label = split_data[data.edge_types[0]].edge_label

    torch.save(label, osp.join(label_dir, 'label.pt'))

    split_idx = get_idx_split(dataset, dataset_name, split_data)

    if dataset_name == 'MovieLens':
        save_link_partitions(split_idx, data, dataset_name, num_parts,
                             save_dir)
    else:
        save_partitions(split_idx, dataset_name, num_parts, save_dir)


def get_dataset(name, dataset_dir, use_sparse_tensor=False):
    transforms = []
    if use_sparse_tensor:
        transforms = [T.ToSparseTensor(remove_edge_index=False)]

    if name == 'ogbn-mag':
        transforms = [T.ToUndirected(merge=True)] + transforms
        return OGB_MAG(
            root=dataset_dir,
            preprocess='metapath2vec',
            transform=T.Compose(transforms),
        )

    elif name == 'ogbn-products':
        transforms = [T.RemoveDuplicatedEdges()] + transforms
        return PygNodePropPredDataset(
            'ogbn-products',
            root=dataset_dir,
            transform=T.Compose(transforms),
        )

    elif name == 'MovieLens':
        transforms = [T.ToUndirected(merge=True)] + transforms
        return MovieLens(
            root=dataset_dir,
            model_name='all-MiniLM-L6-v2',
            transform=T.Compose(transforms),
        )

    elif name == 'Reddit':
        return Reddit(
            root=dataset_dir,
            transform=T.Compose(transforms),
        )


def get_idx_split(dataset, dataset_name, split_data):
    if dataset_name == 'ogbn-mag' or dataset_name == 'Reddit':
        train_idx = mask_to_index(split_data.train_mask)
        test_idx = mask_to_index(split_data.test_mask)
        val_idx = mask_to_index(split_data.val_mask)

    elif dataset_name == 'ogbn-products':
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        test_idx = split_idx['test']
        val_idx = split_idx['valid']

    elif dataset_name == 'MovieLens':
        # Perform a 80/10/10 temporal link-level split:
        perm = torch.argsort(dataset[0][('user', 'rates', 'movie')].time)
        train_idx = perm[:int(0.8 * perm.size(0))]
        val_idx = perm[int(0.8 * perm.size(0)):int(0.9 * perm.size(0))]
        test_idx = perm[int(0.9 * perm.size(0)):]

    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


def save_partitions(split_idx, dataset_name, num_parts, save_dir):
    for key, idx in split_idx.items():
        print(f'-- Partitioning {key} indices ...')
        idx = idx.split(idx.size(0) // num_parts)

        part_dir = osp.join(save_dir, f'{dataset_name}-{key}-partitions')
        os.makedirs(part_dir, exist_ok=True)
        for i in range(num_parts):
            torch.save(idx[i], osp.join(part_dir, f'partition{i}.pt'))


def save_link_partitions(split_idx, data, dataset_name, num_parts, save_dir):
    edge_type = data.edge_types[0]

    for key, idx in split_idx.items():
        print(f'-- Partitioning {key} indices ...')
        edge_index = data[edge_type].edge_index[:, idx]
        edge_index = edge_index.split(edge_index.size(1) // num_parts, dim=1)

        label = data[edge_type].edge_label[idx]
        label = label.split(label.size(0) // num_parts)

        edge_time = data[edge_type].time[idx]
        edge_time = edge_time.split(edge_time.size(0) // num_parts)

        part_dir = osp.join(save_dir, f'{dataset_name}-{key}-partitions')
        os.makedirs(part_dir, exist_ok=True)
        for i in range(num_parts):
            partition = {
                'edge_label_index': edge_index[i],
                'edge_label': label[i],
                'edge_label_time': edge_time[i] - 1,
            }
            torch.save(partition, osp.join(part_dir, f'partition{i}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('--dataset', type=str,
        choices=['ogbn-mag', 'ogbn-products', 'MovieLens',
                 'Reddit'], default='ogbn-products')
    add('--root_dir', default='../../../data', type=str)
    add('--num_partitions', type=int, default=2)
    add('--recursive', action='store_true')
    # TODO (kgajdamo) Add support for arguments below:
    # add('--use-sparse-tensor', action='store_true')
    # add('--bf16', action='store_true')
    args = parser.parse_args()

    partition_dataset(args.dataset, args.root_dir, args.num_partitions,
                      args.recursive)
