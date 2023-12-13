import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures

if torch.cuda.device_count() < 2:
    quit('This example requires multiple GPUs')

path = osp.dirname(osp.realpath(__file__))
path = osp.join(path, '..', '..', 'data', 'Planetoid')
dataset = Planetoid(root=path, name='Cora', transform=NormalizeFeatures())
data = dataset[0].to('cuda:0')


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device1, device2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16).to(device1)
        self.conv2 = GCNConv(16, out_channels).to(device2)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        # Move data to the second device:
        x, edge_index = x.to(self.device2), edge_index.to(self.device2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GCN(
    dataset.num_features,
    dataset.num_classes,
    device1='cuda:0',
    device2='cuda:1',
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index).to('cuda:0')
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index, data.edge_attr)
    pred = out.argmax(dim=-1).to('cuda:0')

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
times = []
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
