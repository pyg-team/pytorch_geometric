import copy
import os
from math import ceil

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(
        self,
        x_all: Tensor,
        device: torch.device,
        subgraph_loader: NeighborLoader,
    ) -> Tensor:
        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                pbar.update(1)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all


def run(world_size: int, rank: int, local_rank: int):
    # Will query the runtime environment for `MASTER_ADDR` and `MASTER_PORT`.
    # Make sure, those are set!
    dist.init_process_group('nccl', world_size=world_size, rank=rank)

    # Download and unzip only with one process ...
    if rank == 0:
        dataset = Reddit('data/Reddit')
    dist.barrier()
    # ... and then read from all the other processes:
    if rank != 0:
        dataset = Reddit('data/Reddit')
    dist.barrier()

    data = dataset[0]

    # Move to device for faster feature fetch.
    data = data.to(local_rank, 'x', 'y')

    # Split training indices into `world_size` many chunks:
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(ceil(train_idx.size(0) / world_size))[rank]

    kwargs = dict(batch_size=1024, num_workers=4, persistent_workers=True)
    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=[25, 10],
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    if rank == 0:  # Create single-hop evaluation neighbor loader:
        subgraph_loader = NeighborLoader(
            copy.copy(data),
            num_neighbors=[-1],
            shuffle=False,
            **kwargs,
        )
        # No need to maintain these features during evaluation:
        del subgraph_loader.data.x, subgraph_loader.data.y
        # Add global node index information:
        subgraph_loader.data.node_id = torch.arange(data.num_nodes)

    torch.manual_seed(12345)
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 21):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x,
                        batch.edge_index.to(local_rank))[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.module.inference(
                    data.x,
                    local_rank,
                    subgraph_loader,
                )
            res = out.argmax(dim=-1) == data.y.to(out.device)
            acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    # Get the world size from the WORLD_SIZE variable or directly from SLURM:
    world_size = int(
        os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    # Likewise for RANK and LOCAL_RANK:
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(
        os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    run(world_size, rank, local_rank)
