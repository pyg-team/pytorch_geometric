import os.path as osp

import torch
import torch.nn as nn
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import DeepGraphInfomax, SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0].to(device, 'x', 'edge_index')

train_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=256,
                              shuffle=True, num_workers=12)
test_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=256,
                             num_workers=12)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
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

    def forward(self, x, edge_index, batch_size):
        for conv, act in zip(self.convs, self.activations):
            x = conv(x, edge_index)
            x = act(x)
        return x[:batch_size]


def corruption(x, edge_index, batch_size):
    return x[torch.randperm(x.size(0))], edge_index, batch_size


model = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(epoch):
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch:02d}'):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(batch.x, batch.edge_index,
                                      batch.batch_size)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pos_z.size(0)
        total_examples += pos_z.size(0)

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    zs = []
    for batch in tqdm(test_loader, desc='Evaluating'):
        pos_z, _, _ = model(batch.x, batch.edge_index, batch.batch_size)
        zs.append(pos_z.cpu())
    z = torch.cat(zs, dim=0)
    train_val_mask = data.train_mask | data.val_mask
    acc = model.test(z[train_val_mask], data.y[train_val_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=10000)
    return acc


for epoch in range(1, 31):
    loss = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
