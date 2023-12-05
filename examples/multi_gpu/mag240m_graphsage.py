import argparse
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from torch import Tensor
from torch.nn import Linear
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader.neighbor_loader import NeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.typing import Adj, EdgeType, NodeType


class HomoGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = x.to(torch.float)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[NodeType], List[EdgeType]],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_nodes_dict: Dict[NodeType, int],
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.5,
        debug: bool = False,
    ):
        super().__init__()
        model = HomoGNN(
            in_channels,
            out_channels,
            hidden_channels,
            num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.model = to_hetero(model, metadata, aggr="sum", debug=debug)
        self.acc = Accuracy(task="multiclass", num_classes=out_channels)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        return self.model(x_dict, edge_index_dict)

    def common_step(self, batch: Batch,
                    ddp_module=None) -> Tuple[Tensor, Tensor]:
        batch_size = batch["paper"].batch_size
        # for node_type, embed in self.embeds.items():
        #     batch[node_type].x = embed(batch[node_type].n_id)
        # # w/o this to_hetero model fails
        for node_type in batch.node_types:
            if node_type not in batch.x_dict.keys():
                paper_x = batch["paper"].x
                # (NOTE) embeddings take too much memory
                batch[node_type].x = torch.zeros(
                    size=(torch.numel(batch[node_type].n_id),
                          paper_x.size(-1)),
                    device=paper_x.device,
                    dtype=paper_x.dtype,
                )
        if ddp_module is None:
            y_hat = self(batch.x_dict,
                         batch.edge_index_dict)["paper"][:batch_size]
        else:
            y_hat = ddp_module(batch.x_dict,
                               batch.edge_index_dict)["paper"][:batch_size]
        y = batch["paper"].y[:batch_size].to(torch.long)
        return y_hat, y

    def training_step(self, batch: Batch, ddp_module=None) -> Tensor:
        y_hat, y = self.common_step(batch, ddp_module)
        train_loss = F.cross_entropy(y_hat, y)
        train_acc = self.acc(y_hat.softmax(dim=-1), y)
        return train_loss, train_acc

    def validation_step(self, batch: Batch):
        y_hat, y = self.common_step(batch)
        return self.acc(y_hat.softmax(dim=-1), y)

    def predict_step(self, batch: Batch):
        y_hat, y = self.common_step(batch)
        return y_hat


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
    debug=False,
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
    model = HeteroGNN(
        data.metadata(),
        data["paper"].x.size(-1),
        data.num_classes,
        hidden_channels,
        num_nodes_dict=data.collect("num_nodes"),
        num_layers=len(sizes),
        dropout=dropout,
        debug=debug,
    )
    if rank == 0:
        print(f"# GNN Params: \
            {sum([p.numel() for p in model.parameters()])/10**6:.1f}M")
        print('Setting up NeighborLoaders...')
    train_idx = data["paper"].train_mask.nonzero(as_tuple=False).view(-1)
    if n_devices > 1:
        # Split training indices into `n_devices` many chunks:
        train_idx = train_idx.split(train_idx.size(0) // n_devices)[rank]
    eval_idx = data["paper"].val_mask.nonzero(as_tuple=False).view(-1)

    kwargs = dict(
        batch_size=batch_size,
        num_workers=get_num_workers(max(n_devices, 1)),
        persistent_workers=True,
        shuffle=True,
        drop_last=True,
    )
    train_loader = NeighborLoader(
        data,
        input_nodes=("paper", train_idx),
        num_neighbors=sizes,
        **kwargs,
    )

    if rank == 0:
        eval_loader = NeighborLoader(
            data,
            input_nodes=("paper", eval_idx),
            num_neighbors=sizes,
            **kwargs,
        )
        test_loader = NeighborLoader(
            data,
            input_nodes=("paper", eval_idx),
            num_neighbors=sizes,
            **kwargs,
        )
    if rank == 0:
        print("Final setup...")
    if n_devices > 0:
        model.to(rank)
    if rank == 0:
        print("about to make optimizer")
    if n_devices > 1:
        ddp = DistributedDataParallel(model, device_ids=[rank])
        optimizer = torch.optim.Adam(ddp.parameters(), lr=0.001)
    else:
        ddp = None
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        model.train()
        time_sum, sum_acc = 0, 0
        if rank == 0:
            print("right before for loop over train loader...")
        if rank == 0 and epoch == 0:
            print("Training beginning...")
        for i, batch in enumerate(train_loader):
            if num_steps_per_epoch >= 0 and i >= num_steps_per_epoch:
                break
            since = time.time()
            optimizer.zero_grad()
            if n_devices > 0:
                batch = batch.to(rank, "x", "y", "edge_index")
            loss, train_acc = model.training_step(batch, ddp)
            sum_acc += train_acc
            loss.backward()
            optimizer.step()
            iter_time = time.time() - since
            if i > num_warmup_iters_for_timing - 1:
                time_sum += iter_time
            if rank == 0 and i % log_every_n_steps == 0:
                print(f"Epoch: {epoch:02d}, Step: {i:d}, Loss: {loss:.4f}, \
                    Train Acc: {sum_acc / (i) * 100.0:.2f}%, \
                    Step Time: {iter_time:.4f}s")
        if n_devices > 1:
            dist.barrier()
        if rank == 0:
            print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, \
                Train Acc:{sum_acc / i * 100.0:.2f}%, \
                Average Step Time: \
                {time_sum/(i - num_warmup_iters_for_timing):.4f}s")
            model.eval()
            acc_sum = 0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    if i >= eval_steps:
                        break
                    if n_devices > 0:
                        batch = batch.to(rank, "x", "y", "edge_index")
                    acc_sum += model.validation_step(batch)
                print(f"Validation Accuracy: {acc_sum/(i + 1) * 100.0:.4f}%")
    if n_devices > 1:
        dist.barrier()
    if rank == 0:
        model.eval()
        acc_sum = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= eval_steps:
                    break
                if n_devices > 0:
                    batch = batch.to(rank, "x", "y", "edge_index")
                acc_sum += model.validation_step(batch)
            print(f"Test Accuracy: {acc_sum/(i + 1) * 100.0:.4f}%", )
    if n_devices > 1:
        dist.destroy_process_group()
    torch.save(model, 'trained_gnn.pt')


def get_num_workers(world_size):
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / (2 * world_size)
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / (2 * world_size)
    return int(num_work)


def estimate_hetero_data_size(data):
    out_bytes = 0
    for n_type in data.node_types:
        for attr_name, n_attr in data.get_node_store(n_type).items():
            if isinstance(n_attr, torch.Tensor) and attr_name != 'x':
                # ignore paper features since their read from disk w/ np memmap
                out_bytes += n_attr.numel() * 64
    for e_type in data.edge_types:
        try:
            out_bytes += data[e_type].edge_index.numel() * 64
        except:  # noqa
            continue
    return out_bytes


if __name__ == "__main__":
    help_str = "-1 by default means run the full \
                    dataset each epoch, \
                    otherwise select how many steps to take."
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_steps_per_epoch", type=int, default=-1,
                        help=help_str)                                                                                                                                                                                                                                                                )
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--num_warmup_iters_for_timing", type=int, default=100)
    parser.add_argument(
        "--subgraph", type=float, default=1,
        help='decimal from (0,1] representing the \
        portion of nodes to use in subgraph')
    parser.add_argument("--sizes", type=str, default="25-15")
    parser.add_argument("--n_devices", type=int, default=1,
                        help="0 devices for CPU, or 1-8 to use GPUs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split("-")]
    print(args)
    if not args.debug:
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
    if not args.evaluate:
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
                        args.debug,
                    ),
                    nprocs=args.n_devices,
                    join=True,
                )
            except ProcessExitedException as e:
                print("torch.multiprocessing.spawn.ProcessExitedException:", e)
                print(
                    "Exceptions/SIGBUS/Errors may be caused by a lack of RAM")

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
                args.debug,
            )
    else:  # eval
        model = torch.load('trained_gnn.pt')
        evaluator = MAG240MEvaluator()
        test_idx = data["paper"].test_mask.nonzero(as_tuple=False).view(-1)
        kwargs = dict(
            batch_size=16,
            num_workers=get_num_workers(1),
            persistent_workers=True,
            shuffle=True,
        )
        test_loader = NeighborLoader(
            data,
            input_nodes=("paper", test_idx),
            num_neighbors=[160] * len(args.sizes),
            **kwargs,
        )
        model.eval()
        device = 'cuda' if (torch.cuda.is_available()
                            and args.n_devices > 0) else 'cpu'
        model.to(device)
        y_preds = []
        print("Evaluating...")
        for batch in tqdm(test_loader):
            batch = batch.to(device, "x", "y", "edge_index")
            with torch.no_grad():
                y_pred = model.predict_step(batch).argmax(dim=-1).cpu()
                y_preds.append(y_pred)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res, 'results/', mode='test-dev')
        print("Done!")
