import argparse
import os.path as osp
import time

import graphlearn_torch as glt
import torch
import torch.distributed
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import GraphSAGE


@torch.no_grad()
def test(model, test_loader, dataset_name):
    evaluator = Evaluator(name=dataset_name)
    model.eval()
    xs = []
    y_true = []
    for i, batch in enumerate(test_loader):
        if i == 0:
            device = batch.x.device
        x = model(batch.x, batch.edge_index)[:batch.batch_size]
        xs.append(x.cpu())
        y_true.append(batch.y[:batch.batch_size].cpu())

    xs = [t.to(device) for t in xs]
    y_true = [t.to(device) for t in y_true]
    y_pred = torch.cat(xs, dim=0).argmax(dim=-1, keepdim=True)
    y_true = torch.cat(y_true, dim=0).unsqueeze(-1)
    test_acc = evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })['acc']
    return test_acc


def run_training_proc(
    local_proc_rank: int,
    num_nodes: int,
    node_rank: int,
    num_training_procs_per_node: int,
    dataset_name: str,
    in_channels: int,
    out_channels: int,
    dataset: glt.distributed.DistDataset,
    train_idx: Tensor,
    test_idx: Tensor,
    epochs: int,
    batch_size: int,
    master_addr: str,
    training_pg_master_port: int,
    train_loader_master_port: int,
    test_loader_master_port: int,
):
    # Initialize graphlearn_torch distributed worker group context:
    glt.distributed.init_worker_group(
        world_size=num_nodes * num_training_procs_per_node,
        rank=node_rank * num_training_procs_per_node + local_proc_rank,
        group_name='distributed-sage-supervised-trainer')

    current_ctx = glt.distributed.get_context()
    current_device = torch.device(local_proc_rank % torch.cuda.device_count())

    # Initialize training process group of PyTorch:
    torch.distributed.init_process_group(
        backend='nccl',  # or choose 'gloo' if 'nccl' is not supported.
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method=f'tcp://{master_addr}:{training_pg_master_port}',
    )

    # Create distributed neighbor loader for training.
    # We replace PyG's NeighborLoader with GLT's DistNeighborLoader.
    # GLT parameters for sampling are quite similar to PyG.
    # We only need to configure additional network and device parameters:
    train_idx = train_idx.split(
        train_idx.size(0) // num_training_procs_per_node)[local_proc_rank]
    train_loader = glt.distributed.DistNeighborLoader(
        data=dataset,
        num_neighbors=[15, 10, 5],
        input_nodes=train_idx,
        batch_size=batch_size,
        shuffle=True,
        collect_features=True,
        to_device=current_device,
        worker_options=glt.distributed.MpDistSamplingWorkerOptions(
            num_workers=1,
            worker_devices=[current_device],
            worker_concurrency=4,
            master_addr=master_addr,
            master_port=train_loader_master_port,
            channel_size='1GB',
            pin_memory=True,
        ),
    )

    # Create distributed neighbor loader for testing.
    test_idx = test_idx.split(test_idx.size(0) //
                              num_training_procs_per_node)[local_proc_rank]
    test_loader = glt.distributed.DistNeighborLoader(
        data=dataset,
        num_neighbors=[15, 10, 5],
        input_nodes=test_idx,
        batch_size=batch_size,
        shuffle=False,
        collect_features=True,
        to_device=current_device,
        worker_options=glt.distributed.MpDistSamplingWorkerOptions(
            num_workers=2,
            worker_devices=[
                torch.device('cuda', i % torch.cuda.device_count())
                for i in range(2)
            ],
            worker_concurrency=4,
            master_addr=master_addr,
            master_port=test_loader_master_port,
            channel_size='2GB',
            pin_memory=True,
        ),
    )

    # Define the model and optimizer.
    torch.cuda.set_device(current_device)
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=256,
        num_layers=3,
        out_channels=out_channels,
    ).to(current_device)
    model = DistributedDataParallel(model, device_ids=[current_device.index])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train and test:
    f = open('dist_sage_sup.txt', 'a+')
    for epoch in range(0, epochs):
        model.train()
        start = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size].long())
            loss.backward()
            optimizer.step()
        f.write(f'-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, '
                f'Loss: {loss:.4f}, Epoch Time: {time.time() - start}\n')

        torch.cuda.synchronize()
        torch.distributed.barrier()

        if epoch == 0 or epoch > (epochs // 2):
            test_acc = test(model, test_loader, dataset_name)
            f.write(f'-- [Trainer {current_ctx.rank}] '
                    f'Test Acc: {test_acc:.4f}\n')
            torch.cuda.synchronize()
            torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='ogbn-products',
        help='The name of the dataset',
    )
    parser.add_argument(
        '--in_channel',
        type=int,
        default=100,
        help='Number of input features of the dataset',
    )
    parser.add_argument(
        '--out_channel',
        type=int,
        default=47,
        help='Number of classes of the dataset',
    )
    parser.add_argument(
        '--num_dataset_partitions',
        type=int,
        default=2,
        help='The number of partitions',
    )
    parser.add_argument(
        '--dataset_root_dir',
        type=str,
        default='../../../data/products',
        help='The root directory (relative path) of the partitioned dataset',
    )
    parser.add_argument(
        '--num_nodes',
        type=int,
        default=2,
        help='Number of distributed nodes',
    )
    parser.add_argument(
        '--node_rank',
        type=int,
        default=0,
        help='The current node rank',
    )
    parser.add_argument(
        '--num_training_procs',
        type=int,
        default=2,
        help='The number of traning processes per node',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='The number of training epochs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='The batch size for the training and testing data loaders',
    )
    parser.add_argument(
        '--master_addr',
        type=str,
        default='localhost',
        help='The master address for RPC initialization',
    )
    parser.add_argument(
        '--training_pg_master_port',
        type=int,
        default=11111,
        help="The port used for PyTorch's process group initialization",
    )
    parser.add_argument(
        '--train_loader_master_port',
        type=int,
        default=11112,
        help='The port used for RPC initialization for training',
    )
    parser.add_argument(
        '--test_loader_master_port',
        type=int,
        default=11113,
        help='The port used for RPC initialization for testing',
    )
    args = parser.parse_args()

    # Record configuration information for debugging
    f = open('dist_sage_sup.txt', 'a+')
    f.write('--- Distributed training example of supervised SAGE ---\n')
    f.write(f'* dataset: {args.dataset}\n')
    f.write(f'* dataset root dir: {args.dataset_root_dir}\n')
    f.write(f'* number of dataset partitions: {args.num_dataset_partitions}\n')
    f.write(f'* total nodes: {args.num_nodes}\n')
    f.write(f'* node rank: {args.node_rank}\n')
    f.write(f'* number of training processes per node: '
            f'{args.num_training_procs}\n')
    f.write(f'* epochs: {args.epochs}\n')
    f.write(f'* batch size: {args.batch_size}\n')
    f.write(f'* master addr: {args.master_addr}\n')
    f.write(f'* training process group master port: '
            f'{args.training_pg_master_port}\n')
    f.write(f'* training loader master port: '
            f'{args.train_loader_master_port}\n')
    f.write(f'* testing loader master port: {args.test_loader_master_port}\n')

    f.write('--- Loading data partition ...\n')
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)),
                        args.dataset_root_dir)
    data_pidx = args.node_rank % args.num_dataset_partitions
    dataset = glt.distributed.DistDataset()

    label_file = osp.join(root_dir, f'{args.dataset}-label', 'label.pt')
    dataset.load(
        root_dir=osp.join(root_dir, f'{args.dataset}-partitions'),
        partition_idx=data_pidx,
        graph_mode='ZERO_COPY',
        whole_node_label_file=label_file,
    )
    train_file = osp.join(root_dir, f'{args.dataset}-train-partitions',
                          f'partition{data_pidx}.pt')
    train_idx = torch.load(train_file)
    test_file = osp.join(root_dir, f'{args.dataset}-test-partitions',
                         f'partition{data_pidx}.pt')
    test_idx = torch.load(test_file)
    train_idx.share_memory_()
    test_idx.share_memory_()

    f.write('--- Launching training processes ...\n')
    torch.multiprocessing.spawn(
        run_training_proc,
        args=(
            args.num_nodes,
            args.node_rank,
            args.num_training_procs,
            args.dataset,
            args.in_channel,
            args.out_channel,
            dataset,
            train_idx,
            test_idx,
            args.epochs,
            args.batch_size,
            args.master_addr,
            args.training_pg_master_port,
            args.train_loader_master_port,
            args.test_loader_master_port,
        ),
        nprocs=args.num_training_procs,
        join=True,
    )
