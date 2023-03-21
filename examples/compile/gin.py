import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GINConv, global_add_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join('data', 'TUDatasets')
transform = T.Pad(max_num_nodes=28, max_num_edges=66)
dataset = TUDataset(path, name='MUTAG', pre_transform=transform).shuffle()

train_dataset = dataset[len(dataset) // 10:]
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          drop_last=True)

test_dataset = dataset[:len(dataset) // 10]
test_loader = DataLoader(test_dataset, batch_size=128)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(5):
            mlp = MLP([in_channels, 32, 32])
            self.convs.append(GINConv(mlp, train_eps=False))
            in_channels = 32

        self.mlp = MLP([32, 32, out_channels], norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


model = GIN(dataset.num_features, dataset.num_classes).to(device)

# Compile the model into an optimized version:
model = torch_geometric.compile(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Test: {test_acc:.4f}')
