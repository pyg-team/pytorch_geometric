import argparse
import os.path as osp
import time
from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.distributed import (
    DistContext,
    DistNeighborLoader,
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.nn import GraphSAGE, to_hetero


@torch.no_grad()
def test(
    model,
    loader,
    dist_context,
    device,
    epoch,
    logfile=None,
    num_loader_threads=10,
    progress_bar=True,
):
    def test_homo(batch):
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y_pred = out.argmax(dim=-1)
        y_true = batch.y[:batch.batch_size]
        return y_pred, y_true

    def test_hetero(batch):
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out['paper'][:batch_size]
        y_pred = out.argmax(dim=-1)
        y_true = batch['paper'].y[:batch_size]
        return y_pred, y_true

    model.eval()
    total_examples = total_correct = 0

    if loader.num_workers > 0:
        context = loader.enable_multithreading(num_loader_threads)
    else:
        context = nullcontext()

    with context:
        if progress_bar:
            loader = tqdm(loader, desc=f'[Node {dist_context.rank}] Test')

        start_time = batch_time = time.time()
        for i, batch in enumerate(loader):
            batch = batch.to(device)

            if isinstance(batch, HeteroData):
                y_pred, y_true = test_hetero(batch)
            else:
                y_pred, y_true = test_homo(batch)

            total_correct += int((y_pred == y_true).sum())
            total_examples += y_pred.size(0)
            batch_acc = int((y_pred == y_true).sum()) / y_pred.size(0)

            result = (f'[Node {dist_context.rank}] Test: epoch={epoch}, '
                      f'it={i}, acc={batch_acc:.4f}, '
                      f'time={(time.time() - batch_time):.4f}')
            batch_time = time.time()

            if logfile:
                log = open(logfile, 'a+')
                log.write(f'{result}\n')
                log.close()

            if not progress_bar:
                print(result)

    total_acc = total_correct / total_examples
    print(f'[Node {dist_context.rank}] Test epoch {epoch} END: '
          f'acc={total_acc:.4f}, time={(time.time() - start_time):.2f}')
    torch.distributed.barrier()


def train(
    model,
    loader,
    optimizer,
    dist_context,
    device,
    epoch,
    logfile=None,
    num_loader_threads=10,
    progress_bar=True,
):
    def train_homo(batch):
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size])
        return loss, batch.batch_size

    def train_hetero(batch):
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out['paper'][:batch_size]
        target = batch['paper'].y[:batch_size]
        loss = F.cross_entropy(out, target)
        return loss, batch_size

    model.train()
    total_loss = total_examples = 0

    if loader.num_workers > 0:
        context = loader.enable_multithreading(num_loader_threads)
    else:
        context = nullcontext()

    with context:
        if progress_bar:
            loader = tqdm(loader, desc=f'[Node {dist_context.rank}] Train')

        start_time = batch_time = time.time()
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            if isinstance(batch, HeteroData):
                loss, batch_size = train_hetero(batch)
            else:
                loss, batch_size = train_homo(batch)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * batch_size
            total_examples += batch_size

            result = (f'[Node {dist_context.rank}] Train: epoch={epoch}, '
                      f'it={i}, loss={loss:.4f}, '
                      f'time={(time.time() - batch_time):.4f}')
            batch_time = time.time()

            if logfile:
                log = open(logfile, 'a+')
                log.write(f'{result}\n')
                log.close()

            if not progress_bar:
                print(result)

    print(f'[Node {dist_context.rank}] Train epoch {epoch} END: '
          f'loss={total_loss/total_examples:.4f}, '
          f'time={(time.time() - start_time):.2f}')
    torch.distributed.barrier()


def run_proc(
    local_proc_rank: int,
    num_nodes: int,
    node_rank: int,
    dataset: str,
    dataset_root_dir: str,
    master_addr: str,
    ddp_port: int,
    train_loader_port: int,
    test_loader_port: int,
    num_epochs: int,
    batch_size: int,
    num_neighbors: str,
    async_sampling: bool,
    concurrency: int,
    num_workers: int,
    num_loader_threads: int,
    progress_bar: bool,
    logfile: str,
):
    is_hetero = dataset == 'ogbn-mag'

    print('--- Loading data partition files ...')
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), dataset_root_dir)
    node_label_file = osp.join(root_dir, f'{dataset}-label', 'label.pt')
    train_idx = torch.load(
        osp.join(
            root_dir,
            f'{dataset}-train-partitions',
            f'partition{node_rank}.pt',
        ))
    test_idx = torch.load(
        osp.join(
            root_dir,
            f'{dataset}-test-partitions',
            f'partition{node_rank}.pt',
        ))

    if is_hetero:
        train_idx = ('paper', train_idx)
        test_idx = ('paper', test_idx)

    # Load partition into local graph store:
    graph = LocalGraphStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)
    # Load partition into local feature store:
    feature = LocalFeatureStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)
    feature.labels = torch.load(node_label_file)
    partition_data = (feature, graph)
    print(f'Partition metadata: {graph.meta}')

    # Initialize distributed context:
    current_ctx = DistContext(
        world_size=num_nodes,
        rank=node_rank,
        global_world_size=num_nodes,
        global_rank=node_rank,
        group_name='distributed-ogb-sage',
    )
    current_device = torch.device('cpu')

    print('--- Initialize DDP training group ...')
    torch.distributed.init_process_group(
        backend='gloo',
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method='tcp://{}:{}'.format(master_addr, ddp_port),
    )

    print('--- Initialize distributed loaders ...')
    num_neighbors = [int(i) for i in num_neighbors.split(',')]
    # Create distributed neighbor loader for training:
    train_loader = DistNeighborLoader(
        data=partition_data,
        input_nodes=train_idx,
        current_ctx=current_ctx,
        device=current_device,
        num_neighbors=num_neighbors,
        shuffle=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=train_loader_port,
        concurrency=concurrency,
        async_sampling=async_sampling,
    )
    # Create distributed neighbor loader for testing:
    test_loader = DistNeighborLoader(
        data=partition_data,
        input_nodes=test_idx,
        current_ctx=current_ctx,
        device=current_device,
        num_neighbors=num_neighbors,
        shuffle=False,
        drop_last=False,
        persistent_workers=num_workers > 0,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=test_loader_port,
        concurrency=concurrency,
        async_sampling=async_sampling,
    )

    print('--- Initialize model ...')
    model = GraphSAGE(
        in_channels=128 if is_hetero else 100,  # num_features
        hidden_channels=256,
        num_layers=len(num_neighbors),
        out_channels=349 if is_hetero else 47,  # num_classes in dataset
    ).to(current_device)

    if is_hetero:  # Turn model into a heterogeneous variant:
        metadata = [
            graph.meta['node_types'],
            [tuple(e) for e in graph.meta['edge_types']],
        ]
        model = to_hetero(model, metadata).to(current_device)
        torch.distributed.barrier()

    # Enable DDP:
    model = DistributedDataParallel(model, find_unused_parameters=is_hetero)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    torch.distributed.barrier()

    # Train and test:
    print(f'--- Start training for {num_epochs} epochs ...')
    for epoch in range(1, num_epochs + 1):
        print(f'Train epoch {epoch}/{num_epochs}:')
        train(
            model,
            train_loader,
            optimizer,
            current_ctx,
            current_device,
            epoch,
            logfile,
            num_loader_threads,
            progress_bar,
        )

        if epoch % 5 == 0:
            print(f'Test epoch {epoch}/{num_epochs}:')
            test(
                model,
                test_loader,
                current_ctx,
                current_device,
                epoch,
                logfile,
                num_loader_threads,
                progress_bar,
            )
    print(f'--- [Node {current_ctx.rank}] Closing ---')
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for distributed training')

    parser.add_argument(
        '--dataset',
        type=str,
        default='ogbn-products',
        choices=['ogbn-products', 'ogbn-mag'],
        help='Name of the dataset: (ogbn-products, ogbn-mag)',
    )
    parser.add_argument(
        '--dataset_root_dir',
        type=str,
        default='../../../data/partitions/ogbn-products/2-parts',
        help='The root directory (relative path) of partitioned dataset',
    )
    parser.add_argument(
        '--num_nodes',
        type=int,
        default=2,
        help='Number of distributed nodes',
    )
    parser.add_argument(
        '--num_neighbors',
        type=str,
        default='15,10,5',
        help='Number of node neighbors sampled at each layer',
    )
    parser.add_argument(
        '--node_rank',
        type=int,
        default=0,
        help='The current node rank',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='The number of training epochs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size for training and testing',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of sampler sub-processes',
    )
    parser.add_argument(
        '--num_loader_threads',
        type=int,
        default=10,
        help='Number of threads used for each sampler sub-process',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=4,
        help='Number of maximum concurrent RPC for each sampler',
    )
    parser.add_argument(
        '--async_sampling',
        type=bool,
        default=True,
        help='Whether sampler processes RPC requests asynchronously',
    )
    parser.add_argument(
        '--master_addr',
        type=str,
        default='localhost',
        help='The master address for RPC initialization',
    )
    parser.add_argument(
        '--ddp_port',
        type=int,
        default=11111,
        help="The port used for PyTorch's DDP communication",
    )
    parser.add_argument(
        '--train_loader_port',
        type=int,
        default=11112,
        help='The port used for RPC communication across training samplers',
    )
    parser.add_argument(
        '--test_loader_port',
        type=int,
        default=11113,
        help='The port used for RPC communication across test samplers',
    )
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--progress_bar', action='store_true')

    args = parser.parse_args()

    print('--- Distributed training example on OGB ---')
    print(f'* total nodes: {args.num_nodes}')
    print(f'* node rank: {args.node_rank}')
    print(f'* dataset: {args.dataset}')
    print(f'* dataset root dir: {args.dataset_root_dir}')
    print(f'* epochs: {args.num_epochs}')
    print(f'* batch size: {args.batch_size}')
    print(f'* number of sampler workers: {args.num_workers}')
    print(f'* master addr: {args.master_addr}')
    print(f'* training process group master port: {args.ddp_port}')
    print(f'* training loader master port: {args.train_loader_port}')
    print(f'* testing loader master port: {args.test_loader_port}')
    print(f'* RPC asynchronous processing: {args.async_sampling}')
    print(f'* RPC concurrency: {args.concurrency}')
    print(f'* loader multithreading: {args.num_loader_threads}')
    print(f'* logging enabled: {args.logging}')
    print(f'* progress bars enabled: {args.progress_bar}')

    if args.logging:
        logfile = f'dist_cpu-node{args.node_rank}.txt'
        with open(logfile, 'a+') as log:
            log.write(f'\n--- Inputs: {str(args)}')
    else:
        logfile = None

    print('--- Launching training processes ...')
    torch.multiprocessing.spawn(
        run_proc,
        args=(
            args.num_nodes,
            args.node_rank,
            args.dataset,
            args.dataset_root_dir,
            args.master_addr,
            args.ddp_port,
            args.train_loader_port,
            args.test_loader_port,
            args.num_epochs,
            args.batch_size,
            args.num_neighbors,
            args.async_sampling,
            args.concurrency,
            args.num_workers,
            args.num_loader_threads,
            args.progress_bar,
            logfile,
        ),
        join=True,
    )
    print('--- Finished training processes ...')
