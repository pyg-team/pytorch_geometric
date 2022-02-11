import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GCNConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.body_convs = torch.nn.ModuleList()
        self.body_norms = torch.nn.ModuleList()

        for layer in range(4):
            in_channels = train_dataset.num_features if layer == 0 else 256
            self.body_convs.append(GCNConv(in_channels, 256))
            self.body_norms.append(BatchNorm(256))

        self.heads = torch.nn.ModuleList()
        for task in range(train_dataset.num_classes):
            self.heads.append(GCNConv(256, 2))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for conv, norm in zip(self.body_convs, self.body_norms):
            x = conv(x, edge_index)
            x = x.relu()
            x = norm(x)
            x = F.dropout(x, 0.5, training=self.training)

        return [head(x, edge_index) for head in self.heads]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        losses = [
            criterion(out, data.y[:, i].long())
            for i, out in enumerate(model(data.x, data.edge_index))
        ]
        loss = sum(losses)
        total_loss += loss.item() / len(losses) * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = [
            torch.unsqueeze(torch.argmax(out, dim=1), dim=1)
            for out in model(data.x.to(device), data.edge_index.to(device))
        ]
        preds.append(torch.cat(out, dim=1).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for epoch in range(1, 101):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
          f'Test: {test_f1:.4f}')
