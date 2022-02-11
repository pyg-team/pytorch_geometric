import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

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

        self.body_conv = GATConv(train_dataset.num_features, 256, heads=4)
        self.body_fc = torch.nn.Linear(train_dataset.num_features, 4 * 256)

        self.head_convs = torch.nn.ModuleList()
        self.head_fc1 = torch.nn.ModuleList()
        self.head_fc2 = torch.nn.ModuleList()
        for task in range(train_dataset.num_classes):
            self.head_convs.append(GATConv(4 * 256, 64, heads=4))
            self.head_fc1.append(torch.nn.Linear(4 * 256, 4 * 64))
            self.head_fc2.append(torch.nn.Linear(4 * 64, 2))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.elu(self.body_conv(x, edge_index) + self.body_fc(x))

        out = []
        for conv, fc1, fc2 in zip(self.head_convs, self.head_fc1,
                                  self.head_fc2):
            x_head = F.elu(conv(x, edge_index) + fc1(x))
            out.append(fc2(x_head))

        return out


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
