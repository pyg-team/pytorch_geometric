import argparse
import os.path as osp
from timeit import default_timer

import tqdm
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader


def run(args: argparse.ArgumentParser) -> None:
    for dataset_name in args.datasets:
        print(f"Dataset: {dataset_name}")
        root = osp.join(args.root, dataset_name)

        if dataset_name == 'mag':
            transform = T.ToUndirected(merge=True)
            dataset = OGB_MAG(root=root, transform=transform)
            train_idx = ('paper', dataset[0]['paper'].train_mask)
            eval_idx = ('paper', None)
            neighbor_sizes = args.hetero_neighbor_sizes
        else:
            dataset = PygNodePropPredDataset(f'ogbn-{dataset_name}', root)
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train']
            eval_idx = None
            neighbor_sizes = args.homo_neighbor_sizes

        data = dataset[0].to(args.device)

        for num_neighbors in neighbor_sizes:
            print(f'Training sampling with {num_neighbors} neighbors')
            for batch_size in args.batch_sizes:
                train_loader = NeighborLoader(
                    data,
                    num_neighbors=num_neighbors,
                    input_nodes=train_idx,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                )
                runtimes = []
                num_iterations = 0
                for run in range(args.runs):
                    start = default_timer()
                    for batch in tqdm.tqdm(train_loader):
                        num_iterations += 1
                    stop = default_timer()
                    runtimes.append(round(stop - start, 3))
                average_time = round(sum(runtimes) / args.runs, 3)
                print(f'batch size={batch_size}, iterations={num_iterations}, '
                      f'runtimes={runtimes}, average runtime={average_time}')

        print('Evaluation sampling with all neighbors')
        for batch_size in args.eval_batch_sizes:
            subgraph_loader = NeighborLoader(
                data,
                num_neighbors=[-1],
                input_nodes=eval_idx,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            runtimes = []
            num_iterations = 0
            for run in range(args.runs):
                start = default_timer()
                for batch in tqdm.tqdm(subgraph_loader):
                    num_iterations += 1
                stop = default_timer()
                runtimes.append(round(stop - start, 3))
            average_time = round(sum(runtimes) / args.runs, 3)
            print(f'batch size={batch_size}, iterations={num_iterations}, '
                  f'runtimes={runtimes}, average runtime={average_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('NeighborLoader Sampling Benchmarking')

    add = parser.add_argument
    add('--device', default='cpu')
    add('--datasets', nargs="+", default=['arxiv', 'products', 'mag'])
    add('--root', default='../../data')
    add('--batch-sizes', default=[8192, 4096, 2048, 1024, 512])
    add('--eval-batch-sizes', default=[16384, 8192, 4096, 2048, 1024, 512])
    add('--homo-neighbor_sizes', default=[[10, 5], [15, 10, 5], [20, 15, 10]])
    add('--hetero-neighbor_sizes', default=[[5], [10], [10, 5]], type=int)
    add('--num-workers', default=0)
    add('--runs', default=3)

    run(parser.parse_args())
