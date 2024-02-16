import argparse
import os.path as osp
import time
from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.distributed import (
    DistContext,
    DistLinkNeighborLoader,
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.nn import SAGEConv, to_hetero


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


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
    model.eval()
    preds, targets = [], []

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

            pred = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch['user', 'movie'].edge_label_index,
            ).clamp(min=0, max=5)
            target = batch['user', 'movie'].edge_label.float()
            preds.append(pred)
            targets.append(target)

            rmse = (pred - target).pow(2).mean().sqrt()

            result = (f'[Node {dist_context.rank}] Test: epoch={epoch}, '
                      f'it={i}, rmse={rmse:.4f}, '
                      f'time={(time.time() - batch_time):.4f}')
            batch_time = time.time()

            if logfile:
                log = open(logfile, 'a+')
                log.write(f'{result}\n')
                log.close()

            if not progress_bar:
                print(result)

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    total_rmse = (pred - target).pow(2).mean().sqrt()
    print(f'[Node {dist_context.rank}] Test epoch {epoch} END: '
          f'rmse={total_rmse:.4f}, time={(time.time() - start_time):.2f}')
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

            pred = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch['user', 'movie'].edge_label_index,
            )
            target = batch['user', 'movie'].edge_label.float()

            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.size(0)
            total_examples += pred.size(0)

            result = (f'[Node {dist_context.rank}] Train: epoch={epoch}, '
                      f'it={i}, loss={loss:.4f}, '
                      f'time={(time.time() - batch_time):.4}')
            batch_time = time.time()

            if logfile:
                log = open(logfile, 'a+')
                log.write(f'{result}\n')
                log.close()

            if not progress_bar:
                print(result)

    torch.distributed.barrier()
    print(f'[Node {dist_context.rank}] Train epoch {epoch} END: '
          f'loss={total_loss/total_examples:.4f}, '
          f'time={(time.time() - start_time):.2f}')


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
    print('--- Loading data partition files ...')
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), dataset_root_dir)
    edge_label_file = osp.join(root_dir, f'{dataset}-label', 'label.pt')
    train_data = torch.load(
        osp.join(
            root_dir,
            f'{dataset}-train-partitions',
            f'partition{node_rank}.pt',
        ))
    test_data = torch.load(
        osp.join(
            root_dir,
            f'{dataset}-test-partitions',
            f'partition{node_rank}.pt',
        ))

    train_edge_label_index = train_data['edge_label_index']
    train_edge_label = train_data['edge_label']
    train_edge_label_time = train_data['edge_label_time']

    test_edge_label_index = test_data['edge_label_index']
    test_edge_label = test_data['edge_label']
    test_edge_label_time = test_data['edge_label_time']

    # Load partition into local graph store:
    graph = LocalGraphStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)
    # Load partition into local feature store:
    feature = LocalFeatureStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)
    feature.labels = torch.load(edge_label_file)
    partition_data = (feature, graph)

    # Add identity user node features for message passing:
    x = torch.eye(
        feature._global_id['user'].size(0),
        feature._feat[('movie', 'x')].size(1),
    )
    feature.put_tensor(x, group_name='user', attr_name='x')

    # Initialize distributed context:
    current_ctx = DistContext(
        world_size=num_nodes,
        rank=node_rank,
        global_world_size=num_nodes,
        global_rank=node_rank,
        group_name='distributed-temporal-link-movielens',
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
    train_loader = DistLinkNeighborLoader(
        data=partition_data,
        edge_label_index=((('user', 'rates', 'movie')),
                          train_edge_label_index),
        edge_label=train_edge_label,
        edge_label_time=train_edge_label_time,
        disjoint=True,
        time_attr='edge_time',
        temporal_strategy='last',
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
    test_loader = DistLinkNeighborLoader(
        data=partition_data,
        edge_label_index=((('user', 'rates', 'movie')), test_edge_label_index),
        edge_label=test_edge_label,
        edge_label_time=test_edge_label_time,
        disjoint=True,
        time_attr='edge_time',
        temporal_strategy='last',
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
    node_types = graph.meta['node_types']
    edge_types = [tuple(e) for e in graph.meta['edge_types']]
    metadata = (node_types, edge_types)
    model = Model(hidden_channels=32, metadata=metadata).to(current_device)

    @torch.no_grad()
    def init_params():  # Init parameters via forwarding a single batch:
        with train_loader as iterator:
            batch = next(iter(iterator))
            batch = batch.to(current_device)
            model(
                batch.x_dict,
                batch.edge_index_dict,
                batch['user', 'movie'].edge_label_index,
            )

    print('--- Initialize parameters of the model ...')
    init_params()
    torch.distributed.barrier()

    # Enable DDP:
    model = DistributedDataParallel(model, find_unused_parameters=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    torch.distributed.barrier()

    # Train and test:
    print(f'--- Start training for {num_epochs} epochs ...')
    for epoch in range(1, num_epochs + 1):
        print(f'Train Epoch {epoch}/{num_epochs}')
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
            print(f'Test Epoch {epoch}/{num_epochs}')
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
        default='MovieLens',
        choices=['MovieLens'],
        help='Name of the dataset: (MovieLens)',
    )
    parser.add_argument(
        '--dataset_root_dir',
        type=str,
        default='../../../data/partitions/MovieLens/2-parts',
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
        default='20,10',
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
        default=1,
        help='Number of max concurrent RPC for each sampler',
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
        help='The port used for PyTorch\'s DDP communication.',
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

    print('--- Distributed training example on MovieLens ---')
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
        logfile = f'dist_cpu-link_temporal{args.node_rank}.txt'
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
