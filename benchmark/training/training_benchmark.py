import argparse
import ast

import torch
import torch.nn.functional as F
from tqdm import tqdm

from benchmark.utils import emit_itt, get_dataset, get_model
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
from torch_geometric.profile import rename_profile_file, timeit, torch_profile

supported_sets = {
    'ogbn-mag': ['rgat', 'rgcn'],
    'ogbn-products': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
    'Reddit': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
}


def train_homo(model, loader, optimizer, device, progress_bar=True,
               desc="") -> None:
    if progress_bar:
        loader = tqdm(loader, desc=desc)
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        if hasattr(batch, 'adj_t'):
            edge_index = batch.adj_t
        else:
            edge_index = batch.edge_index
        out = model(batch.x, edge_index)
        batch_size = batch.batch_size
        out = out[:batch_size]
        target = batch.y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()


def train_hetero(model, loader, optimizer, device, progress_bar=True,
                 desc="") -> None:
    if progress_bar:
        loader = tqdm(loader, desc=desc)
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        if len(batch.adj_t_dict) > 0:
            edge_index_dict = batch.adj_t_dict
        else:
            edge_index_dict = batch.edge_index_dict
        out = model(batch.x_dict, edge_index_dict)
        batch_size = batch['paper'].batch_size
        out = out['paper'][:batch_size]
        target = batch['paper'].y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()


def run(args: argparse.ArgumentParser) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('BENCHMARK STARTS')
    for dataset_name in args.datasets:
        assert dataset_name in supported_sets.keys(
        ), f"Dataset {dataset_name} isn't supported."
        print(f'Dataset: {dataset_name}')
        dataset, num_classes = get_dataset(dataset_name, args.root,
                                           args.use_sparse_tensor, args.bf16)
        data = dataset.to(device)
        hetero = True if dataset_name == 'ogbn-mag' else False
        mask = ('paper', data['paper'].train_mask
                ) if dataset_name == 'ogbn-mag' else data.train_mask
        degree = None
        if torch.cuda.is_available():
            amp = torch.cuda.amp.autocast(enabled=False)
        else:
            amp = torch.cpu.amp.autocast(enabled=args.bf16)

        inputs_channels = data[
            'paper'].num_features if dataset_name == 'ogbn-mag' \
            else dataset.num_features

        for model_name in args.models:
            if model_name not in supported_sets[dataset_name]:
                print(f'Configuration of {dataset_name} + {model_name} '
                      f'not supported. Skipping.')
                continue
            print(f'Training bench for {model_name}:')

            for batch_size in args.batch_sizes:
                for layers in args.num_layers:
                    num_neighbors = args.num_neighbors
                    if type(num_neighbors) is list:
                        if len(num_neighbors) == 1:
                            num_neighbors = num_neighbors * layers
                    elif type(num_neighbors) is int:
                        num_neighbors = [num_neighbors] * layers

                    assert len(
                        num_neighbors) == layers, \
                        f'''num_neighbors={num_neighbors} lenght
                        != num of layers={layers}'''

                    subgraph_loader = NeighborLoader(
                        data,
                        num_neighbors=num_neighbors,
                        input_nodes=mask,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                    )
                    for hidden_channels in args.num_hidden_channels:
                        print('----------------------------------------------')
                        print(f'Batch size={batch_size}, '
                              f'Layers amount={layers}, '
                              f'Num_neighbors={num_neighbors}, '
                              f'Hidden features size={hidden_channels}, '
                              f'Sparse tensor={args.use_sparse_tensor}')

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
                        model.train()
                        optimizer = torch.optim.Adam(model.parameters(),
                                                     lr=0.001)

                        progress_bar = False if args.no_progress_bar else True
                        train = train_hetero if hetero else train_homo
                        with amp:
                            for _ in range(args.warmup):
                                train(model, subgraph_loader, optimizer,
                                      device, progress_bar=progress_bar,
                                      desc="Warmup")
                            with timeit(avg_time_divisor=args.num_epochs):
                                # becomes a no-op if vtune_profile == False
                                with emit_itt(args.vtune_profile):
                                    for epoch in range(args.num_epochs):
                                        train(model, subgraph_loader,
                                              optimizer, device,
                                              progress_bar=progress_bar,
                                              desc=f"Epoch={epoch}")

                            if args.profile:
                                with torch_profile():
                                    train(model, subgraph_loader, optimizer,
                                          device, progress_bar=progress_bar,
                                          desc="Profile training")
                                rename_profile_file(model_name, dataset_name,
                                                    str(batch_size),
                                                    str(layers),
                                                    str(hidden_channels),
                                                    str(num_neighbors))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN training benchmark')
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
    argparser.add_argument('--batch-sizes', nargs='+',
                           default=[512, 1024, 2048, 4096, 8192], type=int)
    argparser.add_argument('--num-layers', nargs='+', default=[2, 3], type=int)
    argparser.add_argument('--num-hidden-channels', nargs='+',
                           default=[64, 128, 256], type=int)
    argparser.add_argument(
        '--num-heads', default=2, type=int,
        help='number of hidden attention heads, applies only for gat and rgat')
    argparser.add_argument('--num-neighbors', default=[10],
                           type=ast.literal_eval,
                           help='number of neighbors to sample per layer')
    argparser.add_argument('--num-workers', default=2, type=int)
    argparser.add_argument('--warmup', default=1, type=int)
    argparser.add_argument('--profile', action='store_true')
    argparser.add_argument('--vtune-profile', action='store_true')
    argparser.add_argument('--bf16', action='store_true')
    argparser.add_argument('--no-progress-bar', action='store_true',
                           default=False, help='turn off using progress bar')
    argparser.add_argument('--num-epochs', default=1, type=int)

    args = argparser.parse_args()

    run(args)
