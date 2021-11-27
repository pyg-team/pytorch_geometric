import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.loader import ClusterData, ClusterLoader
from sklearn.metrics import f1_score

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

train_data = Batch.from_data_list(train_dataset)
cluster_data = ClusterData(train_data, num_parts=50, recursive=False,
                           save_dir=train_dataset.processed_dir)
train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True,
                             num_workers=12)

val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        return self.convs[-1](x, edge_index)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(in_channels=train_dataset.num_features, hidden_channels=1024,
            out_channels=train_dataset.num_classes, num_layers=6).to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
    return total_loss / train_data.num_nodes


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for epoch in range(1, 201):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
          f'Test: {test_f1:.4f}')
