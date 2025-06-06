import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.lsc import MAG240MDataset
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import BatchNorm, HeteroConv, SAGEConv


def common_step(batch, model):
    batch_size = batch['paper'].batch_size
    x_dict = model(batch.x_dict, batch.edge_index_dict)
    y_hat = x_dict['paper'][:batch_size]
    y = batch['paper'].y[:batch_size].to(torch.long)
    return y_hat, y


def training_step(batch, acc, model):
    y_hat, y = common_step(batch, model)
    train_loss = F.cross_entropy(y_hat, y)
    acc(y_hat, y)
    return train_loss


def validation_step(batch, acc, model):
    y_hat, y = common_step(batch, model)
    acc(y_hat, y)


class HeteroSAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, node_types,
                 edge_types, is_output_layer=False):
        super().__init__()
        self.conv = HeteroConv({
            edge_type: SAGEConv(in_channels, out_channels)
            for edge_type in edge_types
        })
        if not is_output_layer:
            self.dropout = torch.nn.Dropout(dropout)
            self.norm_dict = torch.nn.ModuleDict({
                node_type:
                BatchNorm(out_channels)
                for node_type in node_types
            })

        self.is_output_layer = is_output_layer

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        if not self.is_output_layer:
            for node_type, x in x_dict.items():
                x = self.dropout(x.relu())
                x = self.norm_dict[node_type](x)
                x_dict[node_type] = x
        return x_dict


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels,
                 dropout, node_types, edge_types):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            # Since authors and institution do not come with features, we learn
            # them via the GNN. However, this also means we need to exclude
            # them as source types in the first two iterations:
            if i == 0:
                edge_types_of_layer = [
                    edge_type for edge_type in edge_types
                    if edge_type[0] == 'paper'
                ]
            elif i == 1:
                edge_types_of_layer = [
                    edge_type for edge_type in edge_types
                    if edge_type[0] != 'institution'
                ]
            else:
                edge_types_of_layer = edge_types

            conv = HeteroSAGEConv(
                in_channels if i == 0 else hidden_channels,
                out_channels if i == num_layers - 1 else hidden_channels,
                dropout=dropout,
                node_types=node_types,
                edge_types=edge_types_of_layer,
                is_output_layer=i == num_layers - 1,
            )
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return x_dict


def run(
    rank,
    data,
    num_devices,
    num_epochs,
    num_steps_per_epoch,
    log_every_n_steps,
    batch_size,
    num_neighbors,
    hidden_channels,
    dropout,
    num_val_steps,
    lr,
):
    if num_devices > 1:
        if rank == 0:
            print("Setting up distributed...")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=num_devices)

    acc = Accuracy(task='multiclass', num_classes=data.num_classes)
    model = HeteroGraphSAGE(
        in_channels=-1,
        hidden_channels=hidden_channels,
        num_layers=len(num_neighbors),
        out_channels=data.num_classes,
        dropout=dropout,
        node_types=data.node_types,
        edge_types=data.edge_types,
    )

    train_idx = data['paper'].train_mask.nonzero(as_tuple=False).view(-1)
    val_idx = data['paper'].val_mask.nonzero(as_tuple=False).view(-1)
    if num_devices > 1:  # Split indices into `num_devices` many chunks:
        train_idx = train_idx.split(train_idx.size(0) // num_devices)[rank]
        val_idx = val_idx.split(val_idx.size(0) // num_devices)[rank]

    # Delete unused tensors to not sample:
    del data['paper'].train_mask
    del data['paper'].val_mask
    del data['paper'].test_mask
    del data['paper'].year

    kwargs = dict(
        batch_size=batch_size,
        num_workers=16,
        persistent_workers=True,
        num_neighbors=num_neighbors,
        drop_last=True,
    )

    train_loader = NeighborLoader(
        data,
        input_nodes=('paper', train_idx),
        shuffle=True,
        **kwargs,
    )
    val_loader = NeighborLoader(data, input_nodes=('paper', val_idx), **kwargs)

    if num_devices > 0:
        model = model.to(rank)
        acc = acc.to(rank)
    if num_devices > 1:
        model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            if num_steps_per_epoch >= 0 and i >= num_steps_per_epoch:
                break

            if num_devices > 0:
                batch = batch.to(rank, 'x', 'y', 'edge_index')
                # Features loaded in as 16 bits, train in 32 bits:
                batch['paper'].x = batch['paper'].x.to(torch.float32)

            optimizer.zero_grad()
            loss = training_step(batch, acc, model)
            loss.backward()
            optimizer.step()

            if i % log_every_n_steps == 0:
                if rank == 0:
                    print(f"Epoch: {epoch:02d}, Step: {i:d}, "
                          f"Loss: {loss:.4f}, "
                          f"Train Acc: {acc.compute():.4f}")

        if num_devices > 1:
            dist.barrier()

        if rank == 0:
            print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, "
                  f"Train Acc :{acc.compute():.4f}")
        acc.reset()

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                if num_val_steps >= 0 and i >= num_val_steps:
                    break

                if num_devices > 0:
                    batch = batch.to(rank, 'x', 'y', 'edge_index')
                    batch['paper'].x = batch['paper'].x.to(torch.float32)

                validation_step(batch, acc, model)

            if num_devices > 1:
                dist.barrier()

            if rank == 0:
                print(f"Val Acc: {acc.compute():.4f}")
            acc.reset()

    model.eval()

    if num_devices > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_steps_per_epoch", type=int, default=-1)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--num_val_steps", type=int, default=-1, help=50)
    parser.add_argument("--num_neighbors", type=str, default="25-15")
    parser.add_argument("--num_devices", type=int, default=1)
    args = parser.parse_args()

    args.num_neighbors = [int(i) for i in args.num_neighbors.split('-')]

    import warnings
    warnings.simplefilter("ignore")

    if not torch.cuda.is_available():
        args.num_devices = 0
    elif args.num_devices > torch.cuda.device_count():
        args.num_devices = torch.cuda.device_count()

    dataset = MAG240MDataset()
    data = dataset.to_pyg_hetero_data()

    if args.num_devices > 1:
        print("Let's use", args.num_devices, "GPUs!")
        from torch.multiprocessing.spawn import ProcessExitedException
        try:
            mp.spawn(
                run,
                args=(
                    data,
                    args.num_devices,
                    args.num_epochs,
                    args.num_steps_per_epoch,
                    args.log_every_n_steps,
                    args.batch_size,
                    args.num_neighbors,
                    args.hidden_channels,
                    args.dropout,
                    args.num_val_steps,
                    args.lr,
                ),
                nprocs=args.num_devices,
                join=True,
            )
        except ProcessExitedException as e:
            print("torch.multiprocessing.spawn.ProcessExitedException:", e)
            print("Exceptions/SIGBUS/Errors may be caused by a lack of RAM")

    else:
        run(
            0,
            data,
            args.num_devices,
            args.num_epochs,
            args.num_steps_per_epoch,
            args.log_every_n_steps,
            args.batch_size,
            args.num_neighbors,
            args.hidden_channels,
            args.dropout,
            args.num_val_steps,
            args.lr,
        )
