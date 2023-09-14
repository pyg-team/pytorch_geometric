import os.path as osp
import time

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphMultisetTransformer

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PROTEINS')
dataset = TUDataset(path, name='PROTEINS').shuffle()

n = (len(dataset) + 9) // 10
train_dataset = dataset[2 * n:]
val_dataset = dataset[n:2 * n]
test_dataset = dataset[:n]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(dataset.num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        self.pool = GraphMultisetTransformer(96, k=10, heads=4)

        self.lin1 = Linear(96, 16)
        self.lin2 = Linear(16, dataset.num_classes)

    def forward(self, x0, edge_index, batch):
        x1 = self.conv1(x0, edge_index).relu()
        x2 = self.conv2(x1, edge_index).relu()
        x3 = self.conv3(x2, edge_index).relu()
        x = torch.cat([x1, x2, x3], dim=-1)

        x = self.pool(x, batch)

        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        total_loss += data.num_graphs * float(loss)
        optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
    return total_correct / len(loader.dataset)


times = []
for epoch in range(1, 201):
    start = time.time()
    train_loss = train()
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
