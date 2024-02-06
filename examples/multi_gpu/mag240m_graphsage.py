import argparse
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.lsc import MAG240MDataset
from torch import Tensor
from torch.nn import Embedding, Linear
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

from torch_geometric import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader.neighbor_loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.typing import EdgeType, NodeType


def common_step(batch: Batch, model) -> Tuple[Tensor, Tensor]:
    batch_size = batch["paper"].batch_size
    y_hat = model(batch)["paper"][:batch_size]
    y = batch["paper"].y[:batch_size].to(torch.long)
    return y_hat, y


def training_step(batch: Batch, acc, model) -> Tensor:
    y_hat, y = common_step(batch, model)
    train_loss = F.cross_entropy(y_hat, y)
    train_acc = acc(y_hat.softmax(dim=-1), y)
    return train_loss, train_acc


def validation_step(batch: Batch, acc, model):
    y_hat, y = common_step(batch, model)
    return acc(y_hat.softmax(dim=-1), y)


class SAGEConvLayer(torch.nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        dropout,
        edges,
        nodes,
        output_layer=False,
    ):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.conv = HeteroConv(
            {e_type: SAGEConv(in_feat, out_feat)
             for e_type in edges})
        self.output_layer = output_layer
        if not self.output_layer:
            self.dropout_conv = torch.nn.Dropout(dropout)
            self.activation = torch.nn.ReLU()
            self.normalizations = torch.nn.ModuleDict()
            for node in nodes:
                self.normalizations[node] = BatchNorm(out_feat)

    def forward(self, x_dict, edge_index_dict):
        h = self.conv(x_dict, edge_index_dict)
        if not self.output_layer:
            for node_type in h.keys():
                h[node_type] = self.normalizations[node_type](self.activation(
                    self.dropout_conv(h[node_type])))
        return h


# Data = HeteroData(
#   num_classes=153,
#   paper={
#     x=[121751666, 768],
#     y=[121751666],
#     year=[121751666],
#     train_mask=[121751666],
#     val_mask=[121751666],
#     test_mask=[121751666],
#   },
#   author={ num_nodes=122383112 },
#   institution={ num_nodes=25721 },
#   (author, affiliated_with, institution)={ edge_index=[2, 44592586] },
#   (institution, rev_affiliated_with, author)={ edge_index=[2, 44592586] },
#   (author, writes, paper)={ edge_index=[2, 386022720] },
#   (paper, rev_writes, author)={ edge_index=[2, 386022720] },
#   (paper, cites, paper)={ edge_index=[2, 1297748926] },
#   (paper, rev_cites, paper)={ edge_index=[2, 1297748926] }
# )


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels,
                 dropout, metadata):
        super().__init__()
        self.num_layers = num_layers
        self.metadata = metadata
        # only use edges where 'paper' is the source
        # this will propogate info from paper to other node types
        self.in_conv = SAGEConvLayer(
            in_channels, in_channels, dropout,
            [e for e in self.metadata[1] if e[0] == 'paper'], self.metadata[0])
        # only use edges where 'institution' is not the source
        # `institution` still has no features, learn them from `author`/`paper`
        self.in_conv2 = SAGEConvLayer(
            in_channels, hidden_channels, dropout,
            [e for e in self.metadata[1] if e[0] != 'institution'],
            self.metadata[0])
        self.hidden_convs = []
        if self.num_layers > 2:
            for i in range(num_layers - 2):
                self.hidden_convs.append(
                    SAGEConvLayer(hidden_channels, hidden_channels, dropout,
                                  self.metadata[1], self.metadata[0]))
        self.output_conv = SAGEConvLayer(hidden_channels, out_channels,
                                         dropout, self.metadata[1],
                                         self.metadata[0], output_layer=True)

    def forward(self, batch):
        x_dict = batch.collect('x')
        edge_index_dict = batch.collect('edge_index')
        x_dict = self.in_conv(x_dict, edge_index_dict)
        x_dict = self.in_conv2(x_dict, edge_index_dict)
        if self.num_layers > 2:
            for i in range(num_layers - 2):
                x_dict = self.hidden_convs[i](x_dict, edge_index_dict)
        x_dict = self.output_conv(x_dict, edge_index_dict)
        return x_dict


def run(
    rank,
    data,
    n_devices=1,
    num_epochs=1,
    num_steps_per_epoch=-1,
    log_every_n_steps=1,
    batch_size=1024,
    sizes=[128],
    hidden_channels=1024,
    dropout=0.5,
    eval_steps=100,
    num_warmup_iters_for_timing=10,
    lr=.001,
):
    seed_everything(12345)
    if n_devices > 1:
        if rank == 0:
            print("Setting up distributed...")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=n_devices)
    if rank == 0:
        print("Setting up GNN...")
    acc = Accuracy(task="multiclass", num_classes=data.num_classes)
    in_channels = data["paper"].x.size(-1)
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=len(sizes),
        out_channels=data.num_classes,
        dropout=dropout,
        metadata=data.metadata(),
    )

    if rank == 0:
        print(f"# GNN Params: \
            {sum([p.numel() for p in model.parameters()])/10**6:.1f}M")
        print('Setting up NeighborLoaders...')
    train_idx = data["paper"].train_mask.nonzero(as_tuple=False).view(-1)
    eval_idx = data["paper"].val_mask.nonzero(as_tuple=False).view(-1)
    if n_devices > 1:
        # Split indices into `n_devices` many chunks:
        train_idx = train_idx.split(train_idx.size(0) // n_devices)[rank]
        eval_idx = eval_idx.split(eval_idx.size(0) // n_devices)[rank]

    # delete unused tensors to not sample
    del data["paper"].train_mask
    del data["paper"].val_mask
    del data["paper"].test_mask
    del data["paper"].year

    kwargs = dict(batch_size=batch_size, num_workers=16,
                  persistent_workers=True, num_neighbors=sizes, drop_last=True)
    train_loader = NeighborLoader(
        data,
        input_nodes=("paper", train_idx),
        shuffle=True,
        **kwargs,
    )

    eval_loader = NeighborLoader(
        data,
        input_nodes=("paper", eval_idx),
        shuffle=True,
        **kwargs,
    )

    # original OGB example also tests on eval_idx
    # it saves test_idx for the final hidden test
    # for the OGB competition.
    test_loader = NeighborLoader(
        data,
        input_nodes=("paper", eval_idx),
        **kwargs,
    )

    if rank == 0:
        print("Final setup...")
    if n_devices > 0:
        model = model.to(rank)
        acc = acc.to(rank)
    if rank == 0:
        print("about to make optimizer")
    if n_devices > 1:
        model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if rank == 0:
        print("Beginning loop...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        time_sum, acc_sum = 0, 0
        for i, batch in enumerate(train_loader):
            if num_steps_per_epoch >= 0 and i >= num_steps_per_epoch:
                break
            if i >= num_warmup_iters_for_timing:
                torch.cuda.synchronize()
                since = time.time()
            optimizer.zero_grad()

            if n_devices > 0:
                batch = batch.to(rank, "x", "y", "edge_index")
                # Features loaded in as fp16, train in 32bits
                batch['paper'].x = batch['paper'].x.to(torch.float32)
            loss, train_acc = training_step(batch, acc, model)
            acc_sum += train_acc
            loss.backward()
            optimizer.step()
            if n_devices > 1:
                acc_sum = torch.tensor(float(acc_sum), dtype=torch.float32,
                                       device=rank)
                torch.distributed.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
                num_batches = torch.tensor(float(i + 1), dtype=torch.float32,
                                           device=acc_sum.device)
                dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
            else:
                num_batches = i + 1.0
            if i >= num_warmup_iters_for_timing:
                torch.cuda.synchronize()
                iter_time = time.time() - since
                time_sum += iter_time
                if rank == 0 and i % log_every_n_steps == 0:
                    print(
                        f"Epoch: {epoch:02d}, Step: {i:d}, Loss: {loss:.4f}, \
                        Train Acc: {acc_sum / (num_batches) * 100.0:.2f}%, \
                        Most Recent Step Time: {iter_time:.4f}s")
        if n_devices > 1:
            dist.barrier()
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, \
            Train Acc:{acc_sum / num_batches * 100.0:.2f}%, \
            Average Step Time: \
            {time_sum/(num_batches - num_warmup_iters_for_timing):.4f}s")
        model.eval()
        acc_sum = 0.0
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if eval_steps >= 0 and i >= eval_steps:
                    break
                if n_devices > 0:
                    batch = batch.to(rank, "x", "y", "edge_index")
                    # Features loaded in as fp16, train in 32bits
                    batch['paper'].x = batch['paper'].x.to(torch.float32)
                acc_sum += validation_step(batch, acc, model)

            if n_devices > 1:
                acc_sum = torch.tensor(float(acc_sum), dtype=torch.float32,
                                       device=rank)
                torch.distributed.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
                num_batches = torch.tensor(float(i + 1), dtype=torch.float32,
                                           device=acc_sum.device)
                dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
            else:
                num_batches = i + 1.0

            print(
                f"Validation Accuracy: {acc_sum/(num_batches) * 100.0:.4f}%", )
    if n_devices > 1:
        dist.barrier()
    model.eval()
    acc_sum = 0.0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if n_devices > 0:
                batch = batch.to(rank, "x", "y", "edge_index")
                # Features loaded in as fp16, train in 32bits
                batch['paper'].x = batch['paper'].x.to(torch.float32)
            acc_sum += validation_step(batch, acc, model)

        if n_devices > 1:
            acc_sum = torch.tensor(float(acc_sum), dtype=torch.float32,
                                   device=rank)
            torch.distributed.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
            num_batches = torch.tensor(float(i + 1), dtype=torch.float32,
                                       device=acc_sum.device)
            dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        else:
            num_batches = i + 1.0

        final_test_acc = acc_sum / (num_batches) * 100.0
        print(f"Test Accuracy: {final_test_acc:.4f}%", )
    if n_devices > 1:
        dist.destroy_process_group()
    torch.save(model, 'trained_graphsage_for_mag240m.pt')
    assert final_test_acc >= 68.0


if __name__ == "__main__":
    help_str = "-1 by default means run the full \
                    dataset each epoch, \
                    otherwise select how many steps to take."

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_steps_per_epoch", type=int, default=-1,
                        help=help_str)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=-1, help=50)
    parser.add_argument("--num_warmup_iters_for_timing", type=int, default=100)
    parser.add_argument(
        "--subgraph", type=float, default=1,
        help='decimal from (0,1] representing the \
        portion of nodes to use in subgraph')
    parser.add_argument("--sizes", type=str, default="25-15")
    parser.add_argument("--n_devices", type=int, default=1,
                        help="0 devices for CPU, or 1-8 to use GPUs")
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split("-")]
    print(args)

    import warnings
    warnings.simplefilter("ignore")

    if not torch.cuda.is_available():
        print("No GPUs available, running with CPU")
        args.n_devices = 0
    if args.n_devices > torch.cuda.device_count():
        print(
            args.n_devices,
            "GPUs requested but only",
            torch.cuda.device_count(),
            "GPUs available",
        )
        args.n_devices = torch.cuda.device_count()
    print("Loading Data...")
    dataset = MAG240MDataset()
    data = dataset.to_pyg_hetero_data()
    print("Data =", data)

    if args.subgraph < 1.0:
        print("Making a subgraph of the data to \
            save and reduce hardware requirements...")
        data = data.subgraph({
            n_type:
            torch.randperm(
                data[n_type].num_nodes)[:int(data[n_type].num_nodes *
                                             args.subgraph)]
            for n_type in data.node_types
        })
    if args.n_devices > 1:
        print("Let's use", args.n_devices, "GPUs!")
        from torch.multiprocessing.spawn import ProcessExitedException
        try:
            mp.spawn(
                run,
                args=(
                    data,
                    args.n_devices,
                    args.epochs,
                    args.num_steps_per_epoch,
                    args.log_every_n_steps,
                    args.batch_size,
                    args.sizes,
                    args.hidden_channels,
                    args.dropout,
                    args.eval_steps,
                    args.num_warmup_iters_for_timing,
                    args.lr,
                ),
                nprocs=args.n_devices,
                join=True,
            )
        except ProcessExitedException as e:
            print("torch.multiprocessing.spawn.ProcessExitedException:", e)
            print("Exceptions/SIGBUS/Errors may be caused by a lack of RAM")

    else:
        if args.n_devices == 1:
            print("Using a single GPU")
        else:
            print("Using CPU")
        run(
            0,
            data,
            args.n_devices,
            args.epochs,
            args.num_steps_per_epoch,
            args.log_every_n_steps,
            args.batch_size,
            args.sizes,
            args.hidden_channels,
            args.dropout,
            args.eval_steps,
            args.num_warmup_iters_for_timing,
            args.lr,
        )
