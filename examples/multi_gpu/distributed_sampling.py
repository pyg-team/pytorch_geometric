import os
import os.path as osp
from math import ceil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
    ) -> None:
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
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


@torch.no_grad()
def test(
    loader: NeighborLoader,
    model: DistributedDataParallel,
    rank: int,
) -> Tensor:
    model.eval()
    total_correct = torch.tensor(0, dtype=torch.long, device=rank)
    total_examples = 0
    for batch in loader:
        out = model(batch.x, batch.edge_index.to(rank))
        pred = out[:batch.batch_size].argmax(dim=-1)
        y = batch.y[:batch.batch_size].to(rank)
        total_correct += (pred == y).sum()
        total_examples += batch.batch_size

    return total_correct / total_examples


def run(rank: int, world_size: int, dataset: Reddit) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    data = dataset[0]
    data = data.to(rank, 'x', 'y')  # Move to device for faster feature fetch.

    # Split indices into `world_size` many chunks:
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(ceil(train_idx.size(0) / world_size))[rank]
    val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
    val_idx = val_idx.split(ceil(val_idx.size(0) / world_size))[rank]
    test_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
    test_idx = test_idx.split(ceil(test_idx.size(0) / world_size))[rank]

    kwargs = dict(
        data=data,
        batch_size=1024,
        num_neighbors=[25, 10],
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
    )
    train_loader = NeighborLoader(
        input_nodes=train_idx,
        shuffle=True,
        **kwargs,
    )
    val_loader = NeighborLoader(
        input_nodes=val_idx,
        shuffle=False,
        **kwargs,
    )
    test_loader = NeighborLoader(
        input_nodes=test_idx,
        shuffle=False,
        **kwargs,
    )

    torch.manual_seed(12345)
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 21):
        model.train()
        for batch in tqdm(
                train_loader,
                desc=f'Epoch {epoch:02d}',
                disable=rank != 0,
        ):
            out = model(batch.x, batch.edge_index.to(rank))[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if rank == 0:
            print(f'Epoch {epoch:02d}: Train loss: {loss:.4f}')

        if epoch % 5 == 0:
            train_acc = test(train_loader, model, rank)
            val_acc = test(val_loader, model, rank)
            test_acc = test(test_loader, model, rank)

            if world_size > 1:
                dist.all_reduce(train_acc, op=dist.ReduceOp.AVG)
                dist.all_reduce(val_acc, op=dist.ReduceOp.AVG)
                dist.all_reduce(test_acc, op=dist.ReduceOp.AVG)

            if rank == 0:
                print(f'Train acc: {train_acc:.4f}, '
                      f'Val acc: {val_acc:.4f}, '
                      f'Test acc: {test_acc:.4f}')

    dist.destroy_process_group()


if __name__ == '__main__':
    path = osp.join(
        osp.dirname(__file__),
        '..',
        '..',
        'data',
        'Reddit',
    )
    dataset = Reddit(path)
    world_size = torch.cuda.device_count()
    print("Let's use", world_size, "GPUs!")
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
