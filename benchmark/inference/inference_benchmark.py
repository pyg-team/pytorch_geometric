import argparse

import torch
from utils import get_dataset, get_model

import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
from torch_geometric.profile import rename_profile_file, timeit, torch_profile

supported_sets = {
    'ogbn-mag': ['rgat', 'rgcn'],
    'ogbn-products': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
    'Reddit': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
}


def run(args: argparse.ArgumentParser) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('BENCHMARK STARTS')
    for dataset_name in args.datasets:
        assert dataset_name in supported_sets.keys(
        ), f"Dataset {dataset_name} isn't supported."
        print(f'Dataset: {dataset_name}')
        dataset, num_classes = get_dataset(dataset_name, args.root,
                                           args.use_sparse_tensor)
        data = dataset.to(device)
        hetero = True if dataset_name == 'ogbn-mag' else False
        mask = ('paper', None) if dataset_name == 'ogbn-mag' else None
        degree = None
        if torch.cuda.is_available():
            amp = torch.cuda.amp.autocast(enabled=False)
        else:
            amp = torch.cpu.amp.autocast(enabled=args.bf16)
        dtype = torch.float
        if args.bf16:
            dtype = torch.bfloat16

        inputs_channels = data[
            'paper'].num_features if dataset_name == 'ogbn-mag' \
            else dataset.num_features

        for model_name in args.models:
            if model_name not in supported_sets[dataset_name]:
                print(f'Configuration of {dataset_name} + {model_name} '
                      f'not supported. Skipping.')
                continue
            print(f'Evaluation bench for {model_name}:')

            for batch_size in args.eval_batch_sizes:
                if not hetero:
                    subgraph_loader = NeighborLoader(
                        data,
                        num_neighbors=[-1],  # layer-wise inference
                        input_nodes=mask,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                    )

                for layers in args.num_layers:
                    if hetero:
                        subgraph_loader = NeighborLoader(
                            data,
                            num_neighbors=[args.hetero_num_neighbors] *
                            layers,  # batch-wise inference
                            input_nodes=mask,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                        )

                    for hidden_channels in args.num_hidden_channels:
                        print('----------------------------------------------')
                        print(
                            f'Batch size={batch_size}, '
                            f'Layers amount={layers}, '
                            f'Num_neighbors={subgraph_loader.num_neighbors}, '
                            f'Hidden features size={hidden_channels}')
                        params = {
                            'inputs_channels': inputs_channels,
                            'hidden_channels': hidden_channels,
                            'output_channels': num_classes,
                            'num_heads': args.num_heads,
                            'num_layers': layers,
                        }

                        if model_name == 'pna':
                            if degree is None:
                                degree = PNAConv.get_degree_histogram(
                                    subgraph_loader)
                                print(f'Calculated degree for {dataset_name}.')
                            params['degree'] = degree

                        model = get_model(
                            model_name, params,
                            metadata=data.metadata() if hetero else None)
                        model = model.to(device)
                        model.eval()

                        with amp:
                            for _ in range(args.warmup):
                                model.inference(subgraph_loader, device,
                                                progress_bar=True, dtype=dtype)
                            if args.experimental_mode:
                                with torch_geometric.experimental_mode():
                                    with timeit():
                                        model.inference(
                                            subgraph_loader, device,
                                            progress_bar=True, dtype=dtype)
                            else:
                                with timeit():
                                    model.inference(subgraph_loader, device,
                                                    progress_bar=True,
                                                    dtype=dtype)

                            if args.profile:
                                with torch_profile():
                                    model.inference(subgraph_loader, device,
                                                    progress_bar=True,
                                                    dtype=dtype)
                                rename_profile_file(
                                    model_name, dataset_name, str(batch_size),
                                    str(layers), str(hidden_channels),
                                    str(subgraph_loader.num_neighbors))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN inference benchmark')
    argparser.add_argument('--datasets', nargs='+',
                           default=['ogbn-mag', 'ogbn-products',
                                    'Reddit'], type=str)
    argparser.add_argument(
        '--use-sparse-tensor', action='store_true',
        help='use torch_sparse.SparseTensor as graph storage format')
    argparser.add_argument(
        '--models', nargs='+',
        default=['edge_cnn', 'gat', 'gcn', 'pna', 'rgat', 'rgcn'], type=str)
    argparser.add_argument('--root', default='../../data', type=str,
                           help='relative path to look for the datasets')
    argparser.add_argument('--eval-batch-sizes', nargs='+',
                           default=[512, 1024, 2048, 4096, 8192], type=int)
    argparser.add_argument('--num-layers', nargs='+', default=[2, 3], type=int)
    argparser.add_argument('--num-hidden-channels', nargs='+',
                           default=[64, 128, 256], type=int)
    argparser.add_argument(
        '--num-heads', default=2, type=int,
        help='number of hidden attention heads, applies only for gat and rgat')
    argparser.add_argument(
        '--hetero-num-neighbors', default=10, type=int,
        help='number of neighbors to sample per layer for hetero workloads')
    argparser.add_argument('--num-workers', default=0, type=int)
    argparser.add_argument('--experimental-mode', action='store_true',
                           help='use experimental mode')
    argparser.add_argument('--warmup', default=1, type=int)
    argparser.add_argument('--profile', action='store_true')
    argparser.add_argument('--bf16', action='store_true')

    args = argparser.parse_args()

    run(args)
