from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
import os.path as osp
import argparse
from timeit import default_timer
from torch_geometric.loader import NeighborLoader


def run(args: argparse.ArgumentParser) -> None:

    print("BENCHMARK STARTS")
    for dataset_name in args.datasets:
        print("Dataset: ", dataset_name)

        root = osp.join(osp.dirname(osp.realpath(__file__)),
                        args.root, dataset_name.partition("-")[2])

        if dataset_name == 'ogbn-mag':
            transform = T.ToUndirected(merge=True)
            dataset = OGB_MAG(root=root, transform=transform)
            train_idx = ('paper', dataset[0]['paper'].train_mask)
            eval_idx = ('paper', None)
            neighbor_sizes = args.hetero_neighbor_sizes
        else:
            dataset = PygNodePropPredDataset(dataset_name, root)
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train']
            eval_idx = None
            neighbor_sizes = args.homo_neighbor_sizes

        data = dataset[0].to(args.device)

        print('Train sampling')
        for sizes in neighbor_sizes:
            print(f'Sizes={sizes}')
            for batch_size in args.batch_sizes:
                train_loader = NeighborLoader(data,
                                              num_neighbors=sizes,
                                              input_nodes=train_idx,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,)
                start = default_timer()
                iter = 0
                times = []
                for run in range(args.runs):
                    start = default_timer()
                    for batch in train_loader:
                        iter = iter + 1
                    stop = default_timer()
                    times.append(round(stop - start, 3))
                average_time = round(sum(times) / args.runs, 3)
                print(f'Batch size={batch_size}   iterations={iter}   '
                      + f'times={times}    average_time={average_time}')
        print('Evaluation sampling')
        for batch_size in args.eval_batch_sizes:
            subgraph_loader = NeighborLoader(data,
                                             num_neighbors=[-1],
                                             input_nodes=eval_idx,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,)
            start = default_timer()
            iter = 0
            times = []
            for run in range(args.runs):
                start = default_timer()
                for batch in subgraph_loader:
                    iter = iter + 1
                stop = default_timer()
                times.append(round(stop - start, 3))
            average_time = round(sum(times) / args.runs, 3)
            print(f'Batch size={batch_size}   iterations={iter}   '
                + f'times={times}    average_time={average_time}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('NeighborLoader Sampling Benchmarking')

    argparser.add_argument('--device', default='cpu', type=str)
    argparser.add_argument('--datasets', nargs="+",
                           default=['ogbn-arxiv', 'ogbn-products', 'ogbn-mag'], type=str)
    argparser.add_argument('--root', default='../../data', type=str)
    argparser.add_argument(
        '--batch-sizes', default=[8192, 4096, 2048, 1024, 512], type=int)
    argparser.add_argument('--homo-neighbor_sizes',
                           default=[[10, 5], [15, 10, 5], [20, 15, 10]], type=int)
    argparser.add_argument('--hetero-neighbor_sizes',
                           default=[[5], [10], [10, 5]], type=int)
    argparser.add_argument('--eval-batch-sizes',
                           default=[16384, 8192, 4096, 2048, 1024, 512], type=int)
    argparser.add_argument('--num-workers', default=0, type=int)
    argparser.add_argument('--runs', default=3, type=int)

    args = argparser.parse_args()

    run(args)
