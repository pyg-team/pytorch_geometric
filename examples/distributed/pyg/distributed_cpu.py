import argparse
import os.path as osp
import time
from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import torch_geometric.distributed as pyg_dist
from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.partition import load_partition_info
from torch_geometric.nn import GraphSAGE, to_hetero

import logging
logging.basicConfig(
    format='%(levelname)s:%(process)d:%(message)s', level=logging.DEBUG
)


@torch.no_grad()
def test(
    model,
    test_loader,
    is_hetero,
    dist_context,
    device,
    logfile=None,
    num_loader_threads=10,
    progress_bar=True,
):
    def test_homo(batch):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y_pred = out.argmax(dim=-1)
        y_true = batch.y[:batch.batch_size]
        return y_pred, y_true

    def test_hetero(batch):
        batch = batch.to(device, 'edge_index')
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out['paper'][:batch_size]
        y_pred = out.argmax(dim=-1)
        y_true = batch['paper'].y[:batch_size]
        return y_pred, y_true

    test_fn = test_hetero if is_hetero else test_homo
    model.eval()
    total_examples = total_correct = 0
    # Save result at each iteration
    log = open(logfile, 'a+') if logfile else nullcontext()
    multithreading = (test_loader.enable_multithreading(num_loader_threads)
                      if test_loader.num_workers > 0 else nullcontext()
                      )  # speeds up dataloading on CPU
    with log:
        with multithreading:
            if progress_bar:
                test_loader = tqdm(test_loader,
                                   desc=f'[Node {dist_context.rank}] Test')
            batch_time = time.time()
            for i, batch in enumerate(test_loader):
                y_pred, y_true = test_fn(batch)
                total_correct += int((y_pred == y_true).sum())
                total_examples += y_pred.size(0)
                batch_acc = int((y_pred == y_true).sum()) / y_pred.size(0)
                result = (f'[Node {dist_context.rank}] Test: '
                          f'it={i}, acc={batch_acc:.4}, '
                          f'time={(time.time() - batch_time):.4}')
                batch_time = time.time()
                log.write(f'{result}\n')
                if not progress_bar:
                    print(result)
    torch.distributed.barrier()
    return total_correct / total_examples


def train(
    is_hetero,
    model,
    train_loader,
    optimizer,
    dist_context,
    device,
    logfile=None,
    num_loader_threads=10,
    progress_bar=True,
):
    def train_homo(batch):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        return loss

    def train_hetero(batch):
        batch_size = batch['paper'].batch_size
        batch = batch.to(device, 'edge_index')
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        out = out['paper'][:batch_size]
        target = batch['paper'].y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()
        return loss

    train_fn = train_hetero if is_hetero else train_homo
    # Save result at each iteration
    log = open(logfile, 'a+') if logfile else nullcontext()
    multithreading = (train_loader.enable_multithreading(num_loader_threads)
                      if train_loader.num_workers > 0 else nullcontext())

    with log:
        with multithreading:
            if progress_bar:
                train_loader = tqdm(train_loader,
                                    desc=f'[Node {dist_context.rank}] Train')
            batch_time = time.time()
            for i, batch in enumerate(train_loader):
                loss = train_fn(batch)
                result = (f'[Node {dist_context.rank}] Train: '
                          f'it={i}, loss={loss:.4}, '
                          f'time={(time.time() - batch_time):.4}')
                batch_time = time.time()
                log.write(f'{result}\n')
                if not progress_bar:
                    print(result)

    torch.distributed.barrier()
    return loss


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
    if dataset == 'ogbn-mag':
        is_hetero = True
    elif dataset == 'ogbn-products':
        is_hetero = False
    else:
        raise NotImplementedError(f'This example supports only OGB datasets: '
                                  f'(ogbn-products, ogbn-mag), got {dataset}')

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

    # load partition information
    (
        meta,
        num_partitions,
        partition_idx,
        node_pb,
        edge_pb,
    ) = load_partition_info(osp.join(root_dir, f'{dataset}-partitions'),
                            node_rank)
    print(f'meta={meta}, partition_idx={partition_idx}')
    # load partition into graph
    graph = LocalGraphStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)
    # load partition into feature
    feature = LocalFeatureStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)

    # setup the partition information in LocalGraphStore and LocalFeatureStore
    graph.num_partitions = feature.num_partitions = num_partitions
    graph.partition_idx = feature.partition_idx = partition_idx
    graph.node_pb = feature.node_feat_pb = node_pb
    graph.edge_pb = feature.edge_feat_pb = edge_pb
    graph.meta = feature.meta = meta
    feature.labels = torch.load(node_label_file)
    partition_data = (feature, graph)

    # Initialize distributed context
    current_ctx = DistContext(
        world_size=num_nodes,
        rank=node_rank,
        global_world_size=num_nodes,
        global_rank=node_rank,
        group_name='distributed-sage-supervised-Node',
    )
    current_device = torch.device('cpu')

    print('--- Initialize DDP training group ...')
    # Initialize DDP training process group.
    torch.distributed.init_process_group(
        backend='gloo',
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method='tcp://{}:{}'.format(master_addr, ddp_port),
    )
    num_neighbors = [int(i) for i in num_neighbors.split(',')]
    persistent_workers = (True if num_workers > 0 else False
                          )  # Keep workers RPC alive outside the iterator loop
    print('--- Initialize distributed loaders ...')
    # Create distributed neighbor loader for training
    train_loader = pyg_dist.DistNeighborLoader(
        data=partition_data,
        input_nodes=train_idx,
        current_ctx=current_ctx,
        device=current_device,
        num_neighbors=num_neighbors,
        shuffle=True,
        persistent_workers=persistent_workers,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=train_loader_port,
        concurrency=concurrency,
        async_sampling=async_sampling,
    )
    # Create distributed neighbor loader for testing.
    test_loader = pyg_dist.DistNeighborLoader(
        data=partition_data,
        input_nodes=test_idx,
        current_ctx=current_ctx,
        device=current_device,
        num_neighbors=num_neighbors,
        shuffle=False,
        persistent_workers=persistent_workers,
        batch_size=batch_size,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=test_loader_port,
        concurrency=concurrency,
        async_sampling=async_sampling,
    )

    print('--- Initialize model ...')
    # Define model and optimizer.
    model = GraphSAGE(
        in_channels=128 if is_hetero else 100,  # num_features
        hidden_channels=256,
        num_layers=len(num_neighbors),
        out_channels=349 if is_hetero else 47,  # num_classes in dataset
    ).to(current_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

    @torch.no_grad()
    def init_params():
        # Init parameters via forwarding a single batch to the model:
        print('--- Init parameters of the hetero model ...')
        with train_loader as iterator:
            batch = next(iter(iterator))
            batch = batch.to(current_device, 'edge_index')
            model(batch.x_dict, batch.edge_index_dict)
            del batch
            torch.distributed.barrier()

    if is_hetero:
        # Turn model to hetero and initialize parameters
        metadata = [meta['node_types'], [tuple(e) for e in meta['edge_types']]]
        model = to_hetero(model, metadata).to(current_device)
        init_params()
        torch.distributed.barrier()

    # Enable DDP
    model = DistributedDataParallel(model, find_unused_parameters=is_hetero)
    torch.distributed.barrier()

    # Train and test
    print(f'--- Start training for {num_epochs} epochs ...')
    for i in range(num_epochs):
        start = time.time()
        epoch = i + 1
        print(f'Train Epoch {epoch}/{num_epochs}')
        loss = train(
            is_hetero,
            model,
            train_loader,
            optimizer,
            current_ctx,
            current_device,
            logfile,
            num_loader_threads,
            progress_bar,
        )
        print(f'[Node {current_ctx.rank}] Epoch {epoch}: \
                Train Loss = {loss:.4f}, \
                Train Time = {(time.time() - start):.2f}')

        # Test accuracy.
        if i % 5 == 0:
            start = time.time()
            print(f'Test Epoch {epoch}/{num_epochs}')
            acc = test(
                model,
                test_loader,
                is_hetero,
                current_ctx,
                current_device,
                logfile,
                num_loader_threads,
                progress_bar,
            )
            print(f'[Node {current_ctx.rank}] Epoch {epoch}: '
                  f'Test Accuracy = {acc:.4f}, '
                  f'Test Time = {(time.time() - start):.2f}')
    print(f'--- [Node {current_ctx.rank}] Closing ---')
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for distributed training of supervised SAGE.')
    parser.add_argument(
        '--dataset',
        type=str,
        default='ogbn-products',
        help='Name of ogbn dataset: (ogbn-products, ogbn-mag)',
    )
    parser.add_argument(
        '--dataset_root_dir',
        type=str,
        default='../../../data/partitions/ogbn-products/2-parts',
        help='The root directory (relative path) of partitioned ogbn dataset.',
    )
    parser.add_argument(
        '--num_nodes',
        type=int,
        default=2,
        help='Number of distributed nodes.',
    )
    parser.add_argument(
        '--num_neighbors',
        type=str,
        default='15,10,5',
        help='Number of node neighbors sampled at each layer.',
    )
    parser.add_argument(
        '--node_rank',
        type=int,
        default=0,
        help='The current node rank.',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='The number of training epochs.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size for the training and testing dataloader.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of sampler sub-processes.',
    )
    parser.add_argument(
        '--num_loader_threads',
        type=int,
        default=10,
        help='Number of threads used for each sampler sub-process.',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=10,
        help='Number of max concurrent RPC for each sampler.',
    )
    parser.add_argument(
        '--async_sampling',
        type=bool,
        default=True,
        help='If True, samplers process RPC request asynchronously.',
    )
    parser.add_argument(
        '--master_addr',
        type=str,
        default='localhost',
        help='The master address for RPC initialization.',
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
        help='The port used for RPC comm across train loader samplers.',
    )
    parser.add_argument(
        '--test_loader_port',
        type=int,
        default=11113,
        help='The port used for RPC comm across test loader samplers.',
    )
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--progress_bar', action='store_true')

    args = parser.parse_args()

    print('--- Distributed training example with SAGE and OGB dataset ---')
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
