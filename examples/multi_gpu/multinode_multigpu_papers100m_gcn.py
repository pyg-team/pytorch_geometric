'''

in terminal 1:
srun --overlap -A <slurm_access_group> -p interactive \
    -J <experiment-name> -N 2 -t 02:00:00 --pty bash

in terminal 2:

squeue -u <slurm-unix-account-id>
then
export jobid=<JOBID from SQUEUE>

then

srun -l -N2 --ntasks-per-node=1 --overlap --jobid=$jobid
    --container-image=<image_url> --container-name=cont
    --container-mounts=<data-directory>/ogb-papers100m/:/workspace/dataset true

srun -l -N2 --ntasks-per-node=3 --overlap --jobid=$jobid
    --container-name=cont
    --container-mounts=
    <data-directory>/ogb-papers100m/:/workspace/dataset/
python3 multinode-papers100m-gcn.py --ngpu_per_node 3



Results:

Data = Data(num_nodes=111059956, edge_index=[2, 1615685872],
    x=[111059956, 128], node_year=[111059956, 1], y=[111059956])
Using 6 GPUs...
Beginning training...

Epoch: 0, Iteration: 1570, Loss:
    tensor(2.7372, device='cuda:0', grad_fn=<NllLossBackward0>)
Average Training Iteration Time: 0.0022558025027757116 s/iter
Validation Accuracy: 33.1712%
Average Inference Iteration Time: 0.002441989262174637 s/iter

Epoch: 1, Iteration: 1570, Loss:
    tensor(2.6074, device='cuda:0', grad_fn=<NllLossBackward0>)
Average Training Iteration Time: 0.002187901319104231 s/iter
Validation Accuracy: 32.2733%
Average Inference Iteration Time: 0.002225210835015855 s/iter

Epoch: 2, Iteration: 1570, Loss:
    tensor(2.5593, device='cuda:0', grad_fn=<NllLossBackward0>)
Average Training Iteration Time: 0.002199090496994302 s/iter
Validation Accuracy: 33.9588%
Average Inference Iteration Time: 0.003229572181006499 s/iter
Test Accuracy: 24.5902%

'''

import argparse
import os
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

import torch_geometric

warnings.filterwarnings("ignore")


def pyg_num_work(ngpu_per_node):
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / (2 * ngpu_per_node)
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / (2 * ngpu_per_node)
    return int(num_work)


_LOCAL_PROCESS_GROUP = None


def create_local_process_group(num_workers_per_node):
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    assert world_size % num_workers_per_node == 0

    num_nodes = world_size // num_workers_per_node
    node_rank = rank // num_workers_per_node
    for i in range(num_nodes):
        ranks_on_i = list(
            range(i * num_workers_per_node, (i + 1) * num_workers_per_node))
        pg = dist.new_group(ranks_on_i)
        if i == node_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_local_process_group():
    assert _LOCAL_PROCESS_GROUP is not None
    return _LOCAL_PROCESS_GROUP


def run_train(device, data, world_size, ngpu_per_node, model, epochs,
              batch_size, fan_out, split_idx, num_classes,
              cugraph_data_loader):
    local_group = get_local_process_group()
    loc_id = dist.get_rank(group=local_group)
    rank = torch.distributed.get_rank()
    os.environ['NVSHMEM_SYMMETRIC_SIZE'] = "107374182400"
    if rank == 0:
        print("Data =", data)
        print('Using', nprocs, 'GPUs...')
    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[loc_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=0.0005)
    kwargs = dict(
        num_neighbors=[fan_out, fan_out],
        batch_size=batch_size,
    )
    # Set Up Neighbor Loading
    if cugraph_data_loader:
        import cugraph
        from cugraph_pyg.data import CuGraphStore
        from cugraph_pyg.loader import CuGraphNeighborLoader
        G = {("N", "E", "N"): data.edge_index}
        N = {"N": data.num_nodes}
        fs = cugraph.gnn.FeatureStore(backend="torch")
        fs.add_data(data.x, "N", "x")
        fs.add_data(data.y, "N", "y")
        cugraph_store = CuGraphStore(fs, G, N)
        train_loader = CuGraphNeighborLoader(cugraph_store,
                                             input_nodes=split_idx['train'],
                                             shuffle=True, **kwargs)
        if rank == 0:
            eval_loader = CuGraphNeighborLoader(cugraph_store,
                                                input_nodes=split_idx['valid'],
                                                **kwargs)
            test_loader = CuGraphNeighborLoader(cugraph_store,
                                                input_nodes=split_idx['test'],
                                                **kwargs)
    else:
        from torch_geometric.loader import NeighborLoader
        num_work = pyg_num_work(world_size)
        train_loader = NeighborLoader(data, input_nodes=split_idx['train'],
                                      num_workers=num_work, shuffle=True,
                                      **kwargs)
        if rank == 0:
            eval_loader = NeighborLoader(data, input_nodes=split_idx['valid'],
                                         num_workers=num_work, **kwargs)
            test_loader = NeighborLoader(data, input_nodes=split_idx['test'],
                                         num_workers=num_work, **kwargs)

    eval_steps = 1000
    warmup_steps = 100
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    if rank == 0:
        print("Beginning training...")
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i >= warmup_steps:
                start = time.time()
            batch = batch.to(device)
            batch.y = batch.y.to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
            loss.backward()
            optimizer.step()
            if rank == 0 and i % 10 == 0:
                print("Epoch: " + str(epoch) + ", Iteration: " + str(i) +
                      ", Loss: " + str(loss))
        if rank == 0:
            print("Average Training Iteration Time:",
                  (time.time() - start) / (i - 10), "s/iter")
            acc_sum = 0.0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    if i >= eval_steps:
                        break
                    batch = batch.to(device)
                    batch.y = batch.y.to(torch.long)
                    out = model(batch.x, batch.edge_index)
                    acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                   batch.y[:batch_size])
            print(f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
    if rank == 0:
        acc_sum = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(device)
                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)
                acc_sum += acc(out[:batch_size].softmax(dim=-1),
                               batch.y[:batch_size])
            print(f"Test Accuracy: {acc_sum/(i) * 100.0:.4f}%", )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--fan_out', type=int, default=16)
    parser.add_argument(
        "--ngpu_per_node",
        type=int,
        default="1",
        help="number of GPU(s) for each node for multi-gpu training,",
    )
    parser.add_argument(
        "--use_gat_conv",
        action='store_true',
        help="Wether or not to use GATConv. (Defaults to using GCNConv)",
    )
    parser.add_argument(
        "--n_gat_conv_heads",
        type=int,
        default=4,
        help="If using GATConv, number of attention heads to use",
    )
    parser.add_argument(
        "--cugraph_data_loader",
        action='store_true',
        help="Wether or not to use CuGraph for Neighbor Loading. \
            \nNote that this requires more GPU memory or \
            a reduction in batch_size/fan_out/hidden_channels/num_layers",
    )


    args = parser.parse_args()
    # setup multi node
    torch.distributed.init_process_group("nccl")
    nprocs = dist.get_world_size()
    create_local_process_group(args.ngpu_per_node)
    local_group = get_local_process_group()
    device_id = dist.get_rank(
        group=local_group) if dist.is_initialized() else 0
    torch.cuda.set_device(device_id)
    device = torch.device(device_id)

    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    split_idx = dataset.get_idx_split()

    data = dataset[0]
    data.y = data.y.reshape(-1)
    if args.use_gat_conv:
        model = torch_geometric.nn.models.GAT(dataset.num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              dataset.num_classes,
                                              heads=args.n_gat_conv_heads)
    else:
        model = torch_geometric.nn.models.GCN(dataset.num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              dataset.num_classes)
    run_train(device, data, nprocs, args.ngpu_per_node, model, args.epochs,
              args.batch_size, args.fan_out, split_idx, dataset.num_classes,
              args.cugraph_data_loader)
