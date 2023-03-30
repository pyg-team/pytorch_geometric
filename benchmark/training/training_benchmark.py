import argparse
import ast
import warnings
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm import tqdm

from benchmark.utils import (
    emit_itt,
    get_dataset,
    get_model,
    get_split_masks,
    save_benchmark_data,
    test,
    write_to_csv,
)
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
from torch_geometric.profile import rename_profile_file, timeit, torch_profile

supported_sets = {
    'ogbn-mag': ['rgat', 'rgcn'],
    'ogbn-products': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
    'Reddit': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
}


def train_homo(model, loader, optimizer, device, progress_bar=True, desc="",
               trim=False):
    if progress_bar:
        loader = tqdm(loader, desc=desc)
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        if hasattr(batch, 'adj_t'):
            edge_index = batch.adj_t
        else:
            edge_index = batch.edge_index
        if not trim:
            out = model(batch.x, edge_index)
        else:
            out = model(
                batch.x,
                edge_index,
                num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                num_sampled_edges_per_hop=batch.num_sampled_edges,
            )
        batch_size = batch.batch_size
        out = out[:batch_size]
        target = batch.y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()


def train_hetero(model, loader, optimizer, device, progress_bar=True, desc="",
                 trim=False):
    if trim:
        warnings.warn("Trimming not yet implemented for heterogeneous graphs")

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


def run(args: argparse.ArgumentParser):
    csv_data = defaultdict(list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # If we use a custom number of steps, then we need to use RandomSampler,
    # which already does shuffle.
    shuffle = False if args.num_steps != -1 else True

    print('BENCHMARK STARTS')
    for dataset_name in args.datasets:
        assert dataset_name in supported_sets.keys(
        ), f"Dataset {dataset_name} isn't supported."
        print(f'Dataset: {dataset_name}')
        load_time = timeit() if args.measure_load_time else nullcontext()
        with load_time:
            data, num_classes = get_dataset(dataset_name, args.root,
                                            args.use_sparse_tensor, args.bf16)
        hetero = True if dataset_name == 'ogbn-mag' else False
        mask, val_mask, test_mask = get_split_masks(data, dataset_name)
        degree = None
        if torch.cuda.is_available():
            amp = torch.cuda.amp.autocast(enabled=False)
        else:
            amp = torch.cpu.amp.autocast(enabled=args.bf16)

        inputs_channels = data[
            'paper'].num_features if dataset_name == 'ogbn-mag' \
            else data.num_features

        for model_name in args.models:
            if model_name not in supported_sets[dataset_name]:
                print(f'Configuration of {dataset_name} + {model_name} '
                      f'not supported. Skipping.')
                continue
            print(f'Training bench for {model_name}:')

            for batch_size in args.batch_sizes:
                num_nodes = int(mask[-1].sum()) if hetero else int(mask.sum())
                sampler = torch.utils.data.RandomSampler(
                    range(num_nodes), num_samples=args.num_steps *
                    batch_size) if args.num_steps != -1 else None

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

                    kwargs = {
                        'num_neighbors': num_neighbors,
                        'batch_size': batch_size,
                        'shuffle': shuffle,
                        'num_workers': args.num_workers,
                    }
                    subgraph_loader = NeighborLoader(
                        data,
                        input_nodes=mask,
                        sampler=sampler,
                        filter_per_worker=args.filter_per_worker,
                        **kwargs,
                    )
                    if args.evaluate:
                        val_loader = NeighborLoader(
                            data,
                            input_nodes=val_mask,
                            sampler=None,
                            filter_per_worker=args.filter_per_worker,
                            **kwargs,
                        )
                        test_loader = NeighborLoader(
                            data,
                            input_nodes=test_mask,
                            sampler=None,
                            filter_per_worker=args.filter_per_worker,
                            **kwargs,
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

                        # Define context manager parameters:
                        cpu_affinity = subgraph_loader.enable_cpu_affinity(
                            args.loader_cores
                        ) if args.cpu_affinity else nullcontext()

                        with amp, cpu_affinity:
                            for _ in range(args.warmup):
                                train(
                                    model,
                                    subgraph_loader,
                                    optimizer,
                                    device,
                                    progress_bar=progress_bar,
                                    desc="Warmup",
                                    trim=args.trim,
                                )
                            with timeit(avg_time_divisor=args.num_epochs) as t:
                                # becomes a no-op if vtune_profile == False
                                with emit_itt(args.vtune_profile):
                                    for epoch in range(args.num_epochs):
                                        train(
                                            model,
                                            subgraph_loader,
                                            optimizer,
                                            device,
                                            progress_bar=progress_bar,
                                            desc=f"Epoch={epoch}",
                                            trim=args.trim,
                                        )
                                        if args.evaluate:
                                            # In evaluate, throughput and
                                            # latency are not accurate.
                                            val_acc = test(
                                                model, val_loader, device,
                                                hetero,
                                                progress_bar=progress_bar)
                                            print(
                                                f'Val Accuracy: {val_acc:.4f}')

                            if args.evaluate:
                                test_acc = test(model, test_loader, device,
                                                hetero,
                                                progress_bar=progress_bar)
                                print(f'Test Accuracy: {test_acc:.4f}')

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

                        total_time = t.duration
                        if args.num_steps != -1:
                            total_num_samples = args.num_steps * batch_size
                        else:
                            total_num_samples = num_nodes
                        throughput = total_num_samples / total_time
                        latency = total_time / total_num_samples * 1000
                        print(f'Throughput: {throughput:.3f} samples/s')
                        print(f'Latency: {latency:.3f} ms')

                        save_benchmark_data(csv_data, batch_size, layers,
                                            num_neighbors, hidden_channels,
                                            total_time, model_name,
                                            dataset_name,
                                            args.use_sparse_tensor)
    if args.write_csv:
        write_to_csv(csv_data, training=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN training benchmark')
    add = argparser.add_argument

    add('--datasets', nargs='+',
        default=['ogbn-mag', 'ogbn-products', 'Reddit'], type=str)
    add('--use-sparse-tensor', action='store_true',
        help='use torch_sparse.SparseTensor as graph storage format')
    add('--models', nargs='+',
        default=['edge_cnn', 'gat', 'gcn', 'pna', 'rgat', 'rgcn'], type=str)
    add('--root', default='../../data', type=str,
        help='relative path to look for the datasets')
    add('--batch-sizes', nargs='+', default=[512, 1024, 2048, 4096, 8192],
        type=int)
    add('--num-layers', nargs='+', default=[2, 3], type=int)
    add('--num-hidden-channels', nargs='+', default=[64, 128, 256], type=int)
    add('--num-heads', default=2, type=int,
        help='number of hidden attention heads, applies only for gat and rgat')
    add('--num-neighbors', default=[10], type=ast.literal_eval,
        help='number of neighbors to sample per layer')
    add('--num-workers', default=2, type=int)
    add('--warmup', default=1, type=int)
    add('--profile', action='store_true')
    add('--vtune-profile', action='store_true')
    add('--bf16', action='store_true')
    add('--no-progress-bar', action='store_true', default=False,
        help='turn off using progress bar')
    add('--num-epochs', default=1, type=int)
    add('--num-steps', default=-1, type=int,
        help='number of steps, -1 means iterating through all the data')
    add('--cpu-affinity', action='store_true',
        help="Use DataLoader affinitzation.")
    add('--loader-cores', nargs='+', default=[], type=int,
        help="List of CPU core IDs to use for DataLoader workers.")
    add('--filter-per-worker', action='store_true',
        help='Enable filter-per-worker feature of the dataloader.')
    add('--measure-load-time', action='store_true')
    add('--evaluate', action='store_true')
    add('--write-csv', action='store_true', help='Write benchmark data to csv')
    add('--trim', action='store_true', help="Use `trim_to_layer` optimization")
    args = argparser.parse_args()

    run(args)
