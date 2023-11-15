import argparse
import ast
import os
from time import perf_counter
from typing import Any, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from benchmark.utils import get_dataset, get_model, get_split_masks, test
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv

supported_sets = {
    # TODO (DamianSzwichtenberg): cover heterogeneous cases
    # 'ogbn-mag': ['rgat', 'rgcn'],
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
        num_classes: int, data):
    if args.device == 'xpu':
        import intel_extension_for_pytorch as ipex
        import oneccl_bindings_for_pytorch  # noqa
    else:
        # CUDA
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

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
        # TODO (DamianSzwichtenberg):
        # Provide fix for:
        # RuntimeError: Modules with uninitialized parameters can't be used
        # with `DistributedDataParallel`. Run a dummy forward pass to correctly
        # initialize the modules.
        pass
    model = DDP(model, device_ids=[device])
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.device == 'xpu':
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

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


def get_dist_params() -> Tuple[int, int, str]:
    master_addr = "127.0.0.1"
    master_port = "29500"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    mpi_rank = int(os.environ.get("PMI_RANK", -1))
    mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
    rank = mpi_rank if mpi_world_size > 0 else os.environ.get("RANK", 0)
    world_size = (mpi_world_size if mpi_world_size > 0 else os.environ.get(
        "WORLD_SIZE", 1))

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    init_method = f"tcp://{master_addr}:{master_port}"

    return rank, world_size, init_method


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'GNN distributed (DDP) training benchmark')
    add = argparser.add_argument

    add('--device', choices=['cuda', 'xpu'], default='cuda',
        help='Device to run benchmark on')
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
    add(
        '--n-gpus', default=1, type=int,
        help="Only to be used with CUDA devices. \
        For XPU use mpirun to select number of devices")
    add('--evaluate', action='store_true')

    args = argparser.parse_args()
    assert args.dataset in supported_sets.keys(), \
        f"Dataset {args.dataset} isn't supported."
    data, num_classes = get_dataset(args.dataset, args.root)
    if args.device == 'xpu':
        rank, world_size, init_method = get_dist_params()
        dist.init_process_group(backend="ccl", init_method=init_method,
                                world_size=world_size, rank=rank)
        run(
            rank,
            world_size,
            args,
            num_classes,
        )
    else:
        import torch.multiprocessing as mp
        max_world_size = torch.cuda.device_count()
        chosen_world_size = args.n_gpus
        if chosen_world_size <= max_world_size:
            world_size = chosen_world_size
        else:
            print("User selected", chosen_world_size, "GPUs but only",
                  max_world_size, "GPUs are available")
            world_size = max_world_size
        print('Let\'s use', world_size, 'GPUs!')
        mp.spawn(
            run,
            args=(world_size, args, num_classes, data),
            nprocs=world_size,
            join=True,
        )
