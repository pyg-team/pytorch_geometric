import os.path as osp
import argparse

import torch
from tqdm import tqdm
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from torch_geometric.nn import DimeNet

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=0)
parser.add_argument('--cutoff', type=float, default=5.0)
args = parser.parse_args()


class MyTransform(object):  # k-NN graph and feature and target selection.
    def __call__(self, data):
        dist = (data.pos.view(-1, 1, 3) - data.pos.view(1, -1, 3)).norm(dim=-1)
        dist.fill_diagonal_(float('inf'))
        mask = dist <= args.cutoff
        data.edge_index = mask.nonzero().t()
        data.edge_attr = None  # No need to maintain bond types.
        data.x = data.x[:, :5]  # Just make use of atom types as features.
        data.y = data.y[:, args.target]
        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path, transform=MyTransform()).shuffle()
train_dataset = dataset[:110000]
val_dataset = dataset[110000:120000]
test_dataset = dataset[120000:]

train_loader = DataLoader(train_dataset, 44, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, 44, num_workers=6)
test_loader = DataLoader(test_dataset, 44, num_workers=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DimeNet(in_channels=dataset.num_node_features, hidden_channels=128,
                out_channels=1, num_blocks=6, num_bilinear=8, num_spherical=7,
                num_radial=6, cutoff=args.cutoff).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)


def train(loader):
    model.train()

    total_loss = 0
    pbar = tqdm(total=len(loader))
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.pos, data.edge_index, data.batch)
        loss = (out.squeeze(-1) - data.y).abs().mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pbar.set_description(f'Loss: {loss:.4f}')
        pbar.update()

    pbar.close()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_mae = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.pos, data.edge_index, data.batch)
        total_mae += (out.squeeze(-1) - data.y).abs().sum().item()

    return total_mae / len(loader.dataset)


best_val_mae = test_mae = float('inf')
for epoch in range(1, 501):
    train_mae = train(train_loader)
    val_mae = test(val_loader)
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        test_mae = test(test_loader)
    print(f'Epoch: {epoch:02d}, Train: {train_mae:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')
