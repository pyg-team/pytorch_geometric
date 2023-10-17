"""
Distributed GAT training, targeting XPU devices.
PVC has 2 tiles, each reports itself as a separate
device. DDP approach allows us to employ both tiles.

Additional requirements:
    IPEX (intel_extension_for_pytorch)
    oneCCL (oneccl_bindings_for_pytorch)

    We need to import both these modules, as they extend
    torch module with XPU/oneCCL related functionality.

Run with:
    mpirun -np 2 python distributed_sampling_xpu.py
"""

import copy
import os
import os.path as osp
from typing import Tuple, Union

import intel_extension_for_pytorch  # noqa
import oneccl_bindings_for_pytorch  # noqa
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch import Tensor
from torch.nn import Linear as Lin
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dataset.num_features, hidden_channels,
                                  heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(dataset.num_features, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            x = conv(x, edge_index) + skip(x)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(
        self,
        x_all: Tensor,
        device: Union[str, torch.device],
        subgraph_loader: NeighborLoader,
    ) -> Tensor:
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description("Evaluating")

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index) + self.skips[i](x)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def run(rank: int, world_size: int, dataset: PygNodePropPredDataset):
    device = f"xpu:{rank}"

    split_idx = dataset.get_idx_split()
    split_idx["train"] = (split_idx["train"].split(
        split_idx["train"].size(0) // world_size, dim=0)[rank].clone())
    data = dataset[0].to(device, "x", "y")

    kwargs = dict(batch_size=1024, num_workers=0, pin_memory=True)
    train_loader = NeighborLoader(data, input_nodes=split_idx["train"],
                                  num_neighbors=[10, 10, 5], **kwargs)

    if rank == 0:
        subgraph_loader = NeighborLoader(copy.copy(data), num_neighbors=[-1],
                                         **kwargs)
        evaluator = Evaluator(name="ogbn-products")

    torch.manual_seed(12345)
    model = GAT(dataset.num_features, 128, dataset.num_classes, num_layers=3,
                heads=4).to(device)
    model = DDP(model, device_ids=[device])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 21):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x,
                        batch.edge_index.to(device))[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size].squeeze())
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")

        if rank == 0 and epoch % 5 == 0:  # Evaluation on a single GPU
            model.eval()
            with torch.no_grad():
                out = model.module.inference(data.x, device, subgraph_loader)

            y_true = data.y.to(out.device)
            y_pred = out.argmax(dim=-1, keepdim=True)

            train_acc = evaluator.eval({
                "y_true": y_true[split_idx["train"]],
                "y_pred": y_pred[split_idx["train"]],
            })["acc"]
            val_acc = evaluator.eval({
                "y_true": y_true[split_idx["valid"]],
                "y_pred": y_pred[split_idx["valid"]],
            })["acc"]
            test_acc = evaluator.eval({
                "y_true": y_true[split_idx["test"]],
                "y_pred": y_pred[split_idx["test"]],
            })["acc"]

            print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, "
                  f"Test: {test_acc:.4f}")

        dist.barrier()

    dist.destroy_process_group()


def get_dist_params() -> Tuple[int, int, str]:
    master_addr = "127.0.0.1"
    master_port = "29500"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    mpi_rank = int(os.environ.get("PMI_RANK", -1))
    mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
    rank = mpi_rank if mpi_world_size > 0 else os.environ.get("RANK", 0)
    world_size = (mpi_world_size if mpi_world_size > 0 else os.environ.get(
        "WORLD_SIZE", 1))

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    init_method = f"tcp://{master_addr}:{master_port}"

    return rank, world_size, init_method


if __name__ == "__main__":
    rank, world_size, init_method = get_dist_params()
    dist.init_process_group(backend="ccl", init_method=init_method,
                            world_size=world_size, rank=rank)

    path = osp.join(osp.dirname(osp.realpath(__file__)), "../../data",
                    "ogbn-products")
    dataset = PygNodePropPredDataset("ogbn-products", path)

    run(rank, world_size, dataset)
