# Multi-node, multi-GPU example with WholeGraph feature storage.
# It is recommended that you download the dataset first before running.

# To run, use sbatch
# (i.e. sbatch -N2 -p <partition> -A <account> -J <job name>)
# with the script shown below:
#
# head_node_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#
# (yes || true) | srun -l \
#        --container-image <container image> \
#        --container-mounts "$(pwd):/workspace","/raid:/raid" \
#          torchrun \
#          --nnodes 2 \
#          --nproc-per-node 8 \
#          --rdzv-backend c10d \
#          --rdzv-id 62 \
#          --rdzv-endpoint $head_node_addr:29505 \
#          /workspace/papers100m_gcn_cugraph_multinode.py \
#            --epochs 1 \
#            --dataset ogbn-papers100M \
#            --dataset_root /workspace/datasets

import argparse
import json
import os
import os.path as osp
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cugraph.gnn import (
    cugraph_comms_create_unique_id,
    cugraph_comms_init,
    cugraph_comms_shutdown,
)
from ogb.nodeproppred import PygNodePropPredDataset
from pylibwholegraph.torch.initialize import finalize as wm_finalize
from pylibwholegraph.torch.initialize import init as wm_init
from torch.nn.parallel import DistributedDataParallel

import torch_geometric
from torch_geometric.io import fs

# Allow computation on objects that are larger than GPU memory
# https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#spilling-to-host-memory
os.environ['CUDF_SPILL'] = '1'

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ['RAPIDS_NO_INITIALIZE'] = '1'


def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
        pool_allocator=True,
    )

    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    torch.cuda.set_device(local_rank)

    cugraph_comms_init(rank=global_rank, world_size=world_size, uid=cugraph_id,
                       device=local_rank)

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())


def partition_data(dataset, split_idx, edge_path, feature_path, label_path,
                   meta_path):
    data = dataset[0]

    os.makedirs(edge_path, exist_ok=True)
    for (r, e) in enumerate(data.edge_index.tensor_split(world_size, dim=1)):
        rank_path = osp.join(edge_path, f'rank={r}.pt')
        torch.save(
            e.clone(),
            rank_path,
        )

    os.makedirs(feature_path, exist_ok=True)
    for (r, f) in enumerate(torch.tensor_split(data.x, world_size)):
        rank_path = osp.join(feature_path, f'rank={r}_x.pt')
        torch.save(
            f.clone(),
            rank_path,
        )
    for (r, f) in enumerate(torch.tensor_split(data.y, world_size)):
        rank_path = osp.join(feature_path, f'rank={r}_y.pt')
        torch.save(
            f.clone(),
            rank_path,
        )

    os.makedirs(label_path, exist_ok=True)
    for (d, i) in split_idx.items():
        i_parts = torch.tensor_split(i, world_size)
        for r, i_part in enumerate(i_parts):
            rank_path = osp.join(label_path, f'rank={r}')
            os.makedirs(rank_path, exist_ok=True)
            torch.save(i_part, osp.join(rank_path, f'{d}.pt'))

    meta = dict(
        num_classes=int(dataset.num_classes),
        num_features=int(dataset.num_features),
        num_nodes=int(data.num_nodes),
    )
    with open(meta_path, 'w') as f:
        json.dump(meta, f)


def load_partitioned_data(rank, edge_path, feature_path, label_path, meta_path,
                          wg_mem_type):
    from cugraph_pyg.data import GraphStore, WholeFeatureStore

    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = WholeFeatureStore(memory_type=wg_mem_type)

    with open(meta_path) as f:
        meta = json.load(f)

    split_idx = {}
    for split in ['train', 'test', 'valid']:
        path = osp.join(label_path, f'rank={rank}', f'{split}.pt')
        split_idx[split] = fs.torch_load(path)

    path = osp.join(feature_path, f'rank={rank}_x.pt')
    feature_store['node', 'x', None] = fs.torch_load(path)
    path = osp.join(feature_path, f'rank={rank}_y.pt')
    feature_store['node', 'y', None] = fs.torch_load(path)

    eix = fs.torch_load(osp.join(edge_path, f'rank={rank}.pt'))
    graph_store[dict(
        edge_type=('node', 'rel', 'node'),
        layout='coo',
        is_sorted=False,
        size=(meta['num_nodes'], meta['num_nodes']),
    )] = eix

    return (feature_store, graph_store), split_idx, meta


def run(global_rank, data, split_idx, world_size, device, model, epochs,
        batch_size, fan_out, num_classes, wall_clock_start, num_layers=3):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )
    from cugraph_pyg.loader import NeighborLoader

    ix_train = split_idx['train'].cuda()
    train_loader = NeighborLoader(
        data,
        input_nodes=ix_train,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    ix_val = split_idx['valid'].cuda()
    val_loader = NeighborLoader(
        data,
        input_nodes=ix_val,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    ix_test = split_idx['test'].cuda()
    test_loader = NeighborLoader(
        data,
        input_nodes=ix_test,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=80000,
        **kwargs,
    )

    dist.barrier()

    eval_steps = 1000
    warmup_steps = 20
    dist.barrier()
    torch.cuda.synchronize()

    if global_rank == 0:
        prep_time = time.perf_counter() - wall_clock_start
        print(f"Preparation time: {prep_time:.2f}s")

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                torch.cuda.synchronize()
                start = time.time()

            batch = batch.to(device)
            batch_size = batch.batch_size

            batch.y = batch.y.view(-1).to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
            loss.backward()
            optimizer.step()
            if global_rank == 0 and i % 10 == 0:
                print(f"Epoch: {epoch:02d}, Iteration: {i}, Loss: {loss:.4f}")
        nb = i + 1.0

        if global_rank == 0:
            print(f"Avg Training Iteration Time: "
                  f"{(time.time() - start) / (nb - warmup_steps):.4f} s/iter")

        with torch.no_grad():
            total_correct = total_examples = 0
            for i, batch in enumerate(val_loader):
                if i >= eval_steps:
                    break

                batch = batch.to(device)
                batch_size = batch.batch_size

                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)[:batch_size]

                pred = out.argmax(dim=-1)
                y = batch.y[:batch_size].view(-1).to(torch.long)

                total_correct += int((pred == y).sum())
                total_examples += y.size(0)

            acc_val = total_correct / total_examples
            if global_rank == 0:
                print(f"Validation Accuracy: {acc_val * 100:.2f}%", )

        torch.cuda.synchronize()

    with torch.no_grad():
        total_correct = total_examples = 0
        for batch in test_loader:
            batch = batch.to(device)
            batch_size = batch.batch_size

            batch.y = batch.y.to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch_size]

            pred = out.argmax(dim=-1)
            y = batch.y[:batch_size].view(-1).to(torch.long)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)

        acc_test = total_correct / total_examples
        if global_rank == 0:
            print(f"Test Accuracy: {acc_test * 100:.2f}%", )

    if global_rank == 0:
        total_time = time.perf_counter() - wall_clock_start
        print(f"Total Training Runtime: {total_time - prep_time}s")
        print(f"Total Program Runtime: {total_time}s")

    wm_finalize()
    cugraph_comms_shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--fan_out', type=int, default=30)
    parser.add_argument('--dataset', type=str, default='ogbn-papers100M')
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--skip_partition', action='store_true')
    parser.add_argument('--wg_mem_type', type=str, default='distributed')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    wall_clock_start = time.perf_counter()

    # Set a very high timeout so that PyTorch does not crash while
    # partitioning the data.
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    world_size = dist.get_world_size()
    assert dist.is_initialized()

    global_rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(local_rank)

    print(
        f'Global rank: {global_rank},',
        f'Local Rank: {local_rank},',
        f'World size: {world_size}',
    )

    # Create the uid needed for cuGraph comms
    if global_rank == 0:
        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]
    dist.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]

    init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

    edge_path = osp.join(args.root, f'{args.dataset}_eix_part')
    feature_path = osp.join(args.root, f'{args.dataset}_fea_part')
    label_path = osp.join(args.root, f'{args.dataset}_label_part')
    meta_path = osp.join(args.root, f'{args.dataset}_meta.json')

    # We partition the data to avoid loading it in every worker, which will
    # waste memory and can lead to an out of memory exception.
    # cugraph_pyg.GraphStore and cugraph_pyg.WholeFeatureStore are always
    # constructed from partitions of the edge index and features, respectively,
    # so this works well.
    if not args.skip_partition and global_rank == 0:
        print("Partitioning the data into equal size parts per worker")
        dataset = PygNodePropPredDataset(name=args.dataset, root=args.root)
        split_idx = dataset.get_idx_split()

        partition_data(
            dataset,
            split_idx,
            meta_path=meta_path,
            label_path=label_path,
            feature_path=feature_path,
            edge_path=edge_path,
        )

    dist.barrier()
    print("Loading partitioned data")
    data, split_idx, meta = load_partitioned_data(
        rank=global_rank,
        edge_path=edge_path,
        feature_path=feature_path,
        label_path=label_path,
        meta_path=meta_path,
        wg_mem_type=args.wg_mem_type,
    )
    dist.barrier()

    model = torch_geometric.nn.models.GCN(
        meta['num_features'],
        args.hidden_channels,
        args.num_layers,
        meta['num_classes'],
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    run(global_rank, data, split_idx, world_size, device, model, args.epochs,
        args.batch_size, args.fan_out, meta['num_classes'], wall_clock_start,
        args.num_layers)
