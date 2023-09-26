'''
in terminal 1:
srun --overlap -A <slurm_access_group> -p interactive -J <experiment-name> -N 2 -t 02:00:00 --pty bash

in terminal 2:
squeue -u <slurm-unix-account-id>

then

export jobid=<JOBID from SQUEUE>

then

srun -l -N2 --ntasks-per-node=1 --overlap --jobid=$jobid \
--container-image=<image_url> --container-name=cont \
--container-mounts=<data-directory>/ogb-papers100m/:/workspace/dataset true

srun -l -N2 --ntasks-per-node=3 --overlap --jobid=$jobid \
--container-name=cont \
python3 pyg_multinode_tutorial.py --ngpu_per_node 3

'''

import argparse
import os
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GCN

warnings.filterwarnings("ignore")

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


def run(device, data, world_size, model, epochs, batch_size, fan_out,
              split_idx, num_classes):
    local_group = get_local_process_group()
    loc_id = dist.get_rank(group=local_group)
    rank = torch.distributed.get_rank()
    if rank == 0:
        print("Data =", data)
        print('Using', nprocs, 'GPUs...')
    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[loc_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=0.0005)
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    train_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                  input_nodes=split_idx['train'],
                                  batch_size=batch_size)
    if rank == 0:
        eval_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                     input_nodes=split_idx['valid'],
                                     batch_size=batch_size)
        test_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                     input_nodes=split_idx['test'],
                                     batch_size=batch_size)
    eval_steps = 100
    if rank == 0:
        print("Beginning training...")
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i >= 10:
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
                    if i >= 10:
                        start = time.time()
                    batch = batch.to(device)
                    batch.y = batch.y.to(torch.long)
                    out = model(batch.x, batch.edge_index)
                    acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                   batch.y[:batch_size])
            # We should expect poor Val/Test accuracy's since data is random
            print(f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
            print("Average Inference Iteration Time:",
                  (time.time() - start) / (i - 10), "s/iter")
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
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--fan_out', type=int, default=50)
    parser.add_argument(
        "--ngpu_per_node",
        type=int,
        default="1",
        help="number of GPU(s) for each node for multi-gpu training,",
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

    dataset = FakeDataset(avg_num_nodes=100000)
    data = dataset.data
    num_nodes = data.num_nodes
    rand_id = torch.randperm(num_nodes)

    # 60/20/20 split
    split_idx = {
        'train': rand_id[:int(.6 * num_nodes)],
        'valid': rand_id[int(.6 * num_nodes):int(.8 * num_nodes)],
        'test': rand_id[:int(.8 * num_nodes):],
    }

    model = GCN(dataset.num_features, args.hidden_channels, 2,
                dataset.num_classes)
<<<<<<< HEAD
    run(device, data, nprocs, model, args.epochs, args.batch_size,
              args.fan_out, split_idx, dataset.num_classes)
=======
    run_train(device, data, nprocs, model, args.epochs, args.batch_size,
              args.fan_out, split_idx, dataset.num_classes)
>>>>>>> 4b0d4f3508dfc1ec4260f26c1ab90699f651a6c3
