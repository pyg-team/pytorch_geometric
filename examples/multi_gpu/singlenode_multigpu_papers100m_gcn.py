import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

from torch_geometric.nn import GCNConv, GATConv


def pyg_num_work(world_size):
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / (2 * world_size)
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / (2 * world_size)
    return int(num_work)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 use_gat_conv=False, n_gat_conv_heads=4):
        super().__init__()
        if use_gat_conv:
            self.conv1 = GATConv(in_channels, hidden_channels,
                                 heads=n_gat_conv_heads)
            self.conv2 = GATConv(hidden_channels, out_channels,
                                 heads=n_gat_conv_heads)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def run_train(rank, data, world_size, model, epochs, batch_size, fan_out,
              split_idx, num_classes, cugraph_data_loader):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
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
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(rank)
    if rank == 0:
        print("Beginning training...")
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i >= warmup_steps:
                start = time.time()
            batch = batch.to(rank)
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
                  (time.time() - start) / (i - warmup_steps), "s/iter")
            acc_sum = 0.0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    if i >= eval_steps:
                        break
                    if i >= warmup_steps:
                        start = time.time()
                    batch = batch.to(rank)
                    batch.y = batch.y.to(torch.long)
                    out = model(batch.x, batch.edge_index)
                    acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                   batch.y[:batch_size])
            print(f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
            print("Average Inference Iteration Time:",
                  (time.time() - start) / (i - warmup_steps), "s/iter")
    if rank == 0:
        acc_sum = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(rank)
                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)
                acc_sum += acc(out[:batch_size].softmax(dim=-1),
                               batch.y[:batch_size])
            print(f"Test Accuracy: {acc_sum/(i) * 100.0:.4f}%", )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--fan_out', type=int, default=16)
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
        type=bool,
        action='store_true',
        help="Wether or not to use CuGraph for Neighbor Loading",
    )

    args = parser.parse_args()

    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.y = data.y.reshape(-1)
    model = GNN(dataset.num_features, args.hidden_channels,
                dataset.num_classes, args.use_gat_conv, args.n_gat_conv_heads)
    print("Data =", data)
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        run_train, args=(data, world_size, model, args.epochs, args.batch_size,
                         args.fan_out, split_idx, dataset.num_classes,
                         args.cugraph_data_loader), nprocs=world_size,
        join=True)
