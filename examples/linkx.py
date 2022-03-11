import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import LINKXDataset
from torch_geometric.nn import LINKX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'LINKX')
dataset = LINKXDataset(path, name='Penn94')
data = dataset[0].to(device)

model = LINKX(data.num_nodes, data.num_features, hidden_channels=32,
              out_channels=dataset.num_classes, num_layers=1,
              num_edge_layers=1, num_node_layers=1, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    mask = data.train_mask[:, 0]  # Use the first set of the five masks.
    loss = F.cross_entropy(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    accs = []
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        mask = mask[:, 0]  # Use the first set of the five masks.
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
