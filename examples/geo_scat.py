import argparse
import os.path as osp
import time
from typing import Tuple

import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GeoScatConv
from torch_geometric.seed import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate of the prediction head')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=256)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--legs', action='store_true',
                    help='Use learnable LEGS diffusion scales')
parser.add_argument('--legs_J', type=int, default=4,
                    help='LEGS diffusion depth J (ignored unless --legs)')
parser.add_argument('--scattering_orders', type=int, nargs='+', default=[0, 1],
                    choices=[0, 1, 2], help='Scattering orders to include')
parser.add_argument('--pool', type=str, nargs='+',
                    default=['mean', 'var', 'max'],
                    choices=['mean', 'max', 'min', 'median', 'var'],
                    help='Graph-level pooling ops applied by GeoScatConv')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

seed_everything(args.seed)
device = torch_geometric.device('auto')

init_wandb(
    name=f'GeoScat-{args.dataset}',
    batch_size=args.batch_size,
    hidden_channels=args.hidden_channels,
    dropout=args.dropout,
    lr=args.lr,
    epochs=args.epochs,
    seed=args.seed,
    legs=args.legs,
    legs_J=args.legs_J,
    scattering_orders=args.scattering_orders,
    pool=args.pool,
    device=device,
)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name=args.dataset).shuffle()

train_loader = DataLoader(dataset[:0.8], args.batch_size, shuffle=True)
test_loader = DataLoader(dataset[0.8:], args.batch_size)


class GeoScat(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        scattering_orders: Tuple[int, ...],
        pool: Tuple[str, ...],
        use_legs: bool,
        legs_J: int,
        dropout: float,
    ):
        super().__init__()
        diffusion_scales = 'legs' if use_legs else (0, 1, 2, 4, 8, 16)
        legs_kwargs = {'legs_J': legs_J} if use_legs else None
        self.conv = GeoScatConv(
            in_channels,
            scattering_orders=scattering_orders,
            diffusion_scales=diffusion_scales,
            legs_kwargs=legs_kwargs,
            include_lowpass=True,
            pool=pool,
        )
        # DeepSets head over feature channels: ρ(∑_f φ(h_f)).
        self.phi = MLP(
            [self.conv.out_channels, hidden_channels, hidden_channels],
            norm=None)
        self.rho = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # h: (batch_size, num_features, num_scattering_filters * |pool|)
        h = self.conv(x, edge_index, batch=batch)
        h = self.phi(h).sum(dim=1)
        return self.rho(h)


model = GeoScat(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
    scattering_orders=tuple(args.scattering_orders),
    pool=tuple(args.pool),
    use_legs=args.legs,
    legs_J=args.legs_J,
    dropout=args.dropout,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()

    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


times = []
best_test_acc = 0.0
best_epoch = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
print(f'Best test accuracy: {best_test_acc:.4f} (epoch {best_epoch})')
