import argparse
import os.path as osp

import torch
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet
from torch_geometric.utils import remove_self_loops

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=0)
parser.add_argument('--cutoff', type=float, default=5.0)
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path)

model = SchNet.from_qm9_pretrained('/tmp/SchNet', dataset, target=7)
raise NotImplementedError


class TargetSelectTransform(object):
    def __init__(self, target_id=0):
        self.target_id = target_id

    def __call__(self, data):
        data.y = data.y[:, self.target_id]
        return data


class DistanceTransform(object):
    def __call__(self, data):
        len_ = data.x.shape[0]
        i = torch.arange(len_).view(-1, 1).repeat(1, len_).view(-1)
        j = torch.arange(len_).view(1, -1).repeat(len_, 1).view(-1)
        dist_index = torch.stack([i, j])
        dist_index, _ = remove_self_loops(dist_index)
        data.dist_index = dist_index
        data.dist = (data.pos[dist_index[0]] -
                     data.pos[dist_index[1]]).norm(dim=-1).view(-1, 1)
        return data


transform = T.Compose(
    [TargetSelectTransform(args.target),
     DistanceTransform()])
train_dataset = dataset[:110000]
val_dataset = dataset[110000:120000]
test_dataset = dataset[120000:]

train_loader = DataLoader(train_dataset, 32, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, 32, num_workers=6)
test_loader = DataLoader(test_dataset, 32, num_workers=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SchNet(dataset.num_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(loader):
    model.train()

    total_loss = 0
    pbar = tqdm(total=len(loader))
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
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
        out = model(data)
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
