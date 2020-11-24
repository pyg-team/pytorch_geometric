import os.path as osp

import torch
import torch.nn as nn

from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import DeepGraphInfomax
'''
In '"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>' two methods
were provided, a transductive method for small graphs (see examples/infomax.py)
and a inductive method for large graphs, implemented here. The inductive
implementation utilises the mean pooling propagation rule as detailed in
"Inductive Representation Learning on Large Graphs"
<https://arxiv.org/abs/1706.02216>.
'''

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]

train_loader = NeighborSampler(data.edge_index, node_idx=None,
                               sizes=[10, 10, 25], batch_size=256,
                               shuffle=True, num_workers=12)

test_loader = NeighborSampler(data.edge_index, node_idx=None,
                              sizes=[10, 10, 25], batch_size=256,
                              shuffle=False, num_workers=12)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.extend([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels)
        ])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels)
        ])

    def forward(self, x, adjs):

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()
    total_loss = 0

    for batch_size, n_id, adjs in tqdm(train_loader,
                                       desc=f'Epoch {epoch:02d}'):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()

        pos_z, neg_z, summary = model(x[n_id], adjs)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test():

    model.eval()

    for i, (batch_size, n_id, adjs) in enumerate(test_loader):

        adjs = [adj.to(device) for adj in adjs]
        z, _, _ = model(x[n_id], adjs)

        if i == 0:
            z_test = z
        else:
            z_test = torch.cat([z_test, z], dim=0)

    train_val_mask = torch.cat([data.train_mask, data.val_mask])
    acc = model.test(z_test[train_val_mask], y[train_val_mask],
                     z_test[data.test_mask], y[data.test_mask], max_iter=10000)

    return acc


for epoch in range(1, 150):

    loss = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
