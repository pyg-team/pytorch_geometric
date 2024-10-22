import argparse
import ast
from time import perf_counter
from typing import Any, Callable, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from benchmark.utils import get_model, get_split_masks, test
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv

supported_sets = {
    'ogbn-mag': ['rgat', 'rgcn'],
    'ogbn-products': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
    'Reddit': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
}

device_conditions = {
    'xpu': (lambda: torch.xpu.is_available()),
    'cuda': (lambda: torch.cuda.is_available()),
}


def train_homo(model: Any, loader: NeighborLoader, optimizer: torch.optim.Adam,
               device: torch.device) -> torch.Tensor:
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        batch_size = batch.batch_size
        out = out[:batch_size]
        target = batch.y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

    return loss


def train_hetero(model: Any, loader: NeighborLoader,
                 optimizer: torch.optim.Adam,
                 device: torch.device) -> torch.Tensor:
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        batch_size = batch['paper'].batch_size
        out = out['paper'][:batch_size]
        target = batch['paper'].y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

    return loss


def maybe_synchronize(device: str):
    if device == 'xpu' and torch.xpu.is_available():
        torch.xpu.synchronize()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()


def create_mask_per_rank(
        global_mask: Union[torch.Tensor,
                           Tuple[str,
                                 torch.Tensor]], rank: int, world_size: int,
        hetero: bool = False) -> Union[torch.Tensor, Tuple[str, torch.Tensor]]:
    mask = global_mask[-1] if hetero else global_mask
    nonzero = mask.nonzero().reshape(-1)
    rank_indices = nonzero.split(nonzero.size(0) // world_size,
                                 dim=0)[rank].clone()
    mask_per_rank = torch.full_like(mask, False)
    mask_per_rank[rank_indices] = True

    if hetero:
        return tuple((global_mask[0], mask_per_rank))
    else:
        return mask_per_rank


def run(rank: int, world_size: int, args: argparse.ArgumentParser,
        num_classes: int, data: Union[Data, HeteroData],
        custom_optimizer: Callable[[Any, Any], Tuple[Any, Any]] = None):
    if not device_conditions[args.device]():
        raise RuntimeError(f'{args.device.upper()} is not available')

    device = torch.device(f'{args.device}:{rank}')

    if rank == 0:
        print('BENCHMARK STARTS')
        print(f'Running on {args.device.upper()}')
        print(f'Dataset: {args.dataset}')

    hetero = True if args.dataset == 'ogbn-mag' else False
    mask, val_mask, test_mask = get_split_masks(data, args.dataset)
    mask = create_mask_per_rank(mask, rank, world_size, hetero)
    degree = None

    inputs_channels = data[
        'paper'].num_features if args.dataset == 'ogbn-mag' \
        else data.num_features

    if args.model not in supported_sets[args.dataset]:
        err_msg = (f'Configuration of {args.dataset} + {args.model}'
                   'not supported')
        raise RuntimeError(err_msg)
    if rank == 0:
        print(f'Training bench for {args.model}:')

    num_nodes = int(mask[-1].sum()) if hetero else int(mask.sum())
    num_neighbors = args.num_neighbors

    if type(num_neighbors) is list:
        if len(num_neighbors) == 1:
            num_neighbors = num_neighbors * args.num_layers
    elif type(num_neighbors) is int:
        num_neighbors = [num_neighbors] * args.num_layers

    if len(num_neighbors) != args.num_layers:
        err_msg = (f'num_neighbors={num_neighbors} lenght != num of'
                   'layers={args.num_layers}')

    kwargs = {
        'num_neighbors': num_neighbors,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
    }
    subgraph_loader = NeighborLoader(
        data,
        input_nodes=mask,
        sampler=None,
        **kwargs,
    )
    if rank == 0 and args.evaluate:
        val_loader = NeighborLoader(
            data,
            input_nodes=val_mask,
            sampler=None,
            **kwargs,
        )
        test_loader = NeighborLoader(
            data,
            input_nodes=test_mask,
            sampler=None,
            **kwargs,
        )

    if rank == 0:
        print('----------------------------------------------')
        print(
            f'Batch size={args.batch_size}, '
            f'Layers amount={args.num_layers}, '
            f'Num_neighbors={num_neighbors}, '
            f'Hidden features size={args.num_hidden_channels}', flush=True)

    params = {
        'inputs_channels': inputs_channels,
        'hidden_channels': args.num_hidden_channels,
        'output_channels': num_classes,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
    }

    if args.model == 'pna' and degree is None:
        degree = PNAConv.get_degree_histogram(subgraph_loader)
        print(f'Rank: {rank}, calculated degree for {args.dataset}.',
              flush=True)
        params['degree'] = degree
    dist.barrier()

    torch.manual_seed(12345)
    model = get_model(args.model, params,
                      metadata=data.metadata() if hetero else None)
    model = model.to(device)

    if hetero:
        model.eval()
        x_keys = data.metadata()[0]
        edge_index_keys = data.metadata()[1]
        fake_x_dict = {
            k: torch.rand((32, inputs_channels), device=device)
            for k in x_keys
        }
        fake_edge_index_dict = {
            k: torch.randint(0, 32, (2, 8), device=device)
            for k in edge_index_keys
        }
        model.forward(fake_x_dict, fake_edge_index_dict)

    model = DDP(model, device_ids=[device], find_unused_parameters=hetero)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if custom_optimizer:
        model, optimizer = custom_optimizer(model, optimizer)

    train = train_hetero if hetero else train_homo

    maybe_synchronize(args.device)
    dist.barrier()
    if rank == 0:
        beg = perf_counter()

    for epoch in range(args.num_epochs):
        loss = train(
            model,
            subgraph_loader,
            optimizer,
            device,
        )

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}', flush=True)

        if rank == 0 and args.evaluate:
            # In evaluate, throughput and
            # latency are not accurate.
            val_acc = test(model, val_loader, device, hetero,
                           progress_bar=False)
            print(f'Val Accuracy: {val_acc:.4f}')

        dist.barrier()

    maybe_synchronize(args.device)
    dist.barrier()
    if rank == 0:
        end = perf_counter()
        duration = end - beg

    if rank == 0 and args.evaluate:
        test_acc = test(model, test_loader, device, hetero, progress_bar=False)
        print(f'Test Accuracy: {test_acc:.4f}')

    dist.barrier()

    if rank == 0:
        num_nodes_total = num_nodes * world_size
        duration_per_epoch = duration / args.num_epochs
        throughput = num_nodes_total / duration_per_epoch
        latency = duration_per_epoch / num_nodes_total * 1000
        print(f'Time: {duration_per_epoch:.4f}s')
        print(f'Throughput: {throughput:.3f} samples/s')
        print(f'Latency: {latency:.3f} ms', flush=True)

    dist.destroy_process_group()


def get_predefined_args() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        'GNN distributed (DDP) training benchmark')
    add = argparser.add_argument

    add('--dataset', choices=['ogbn-mag', 'ogbn-products', 'Reddit'],
        default='Reddit', type=str)
    add('--model',
        choices=['edge_cnn', 'gat', 'gcn', 'pna', 'rgat', 'rgcn',
                 'sage'], default='sage', type=str)
    add('--root', default='../../data', type=str,
        help='relative path to look for the datasets')
    add('--batch-size', default=4096, type=int)
    add('--num-layers', default=3, type=int)
    add('--num-hidden-channels', default=128, type=int)
    add('--num-heads', default=2, type=int,
        help='number of hidden attention heads, applies only for gat and rgat')
    add('--num-neighbors', default=[10], type=ast.literal_eval,
        help='number of neighbors to sample per layer')
    add('--num-workers', default=0, type=int)
    add('--num-epochs', default=1, type=int)
    add('--evaluate', action='store_true')

    return argparser
