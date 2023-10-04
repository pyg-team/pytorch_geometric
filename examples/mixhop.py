import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import BatchNorm, Linear, MixHopConv

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, name='Cora')
data = dataset[0]


class MixHop(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MixHopConv(dataset.num_features, 60, powers=[0, 1, 2])
        self.norm1 = BatchNorm(3 * 60)

        self.conv2 = MixHopConv(3 * 60, 60, powers=[0, 1, 2])
        self.norm2 = BatchNorm(3 * 60)

        self.conv3 = MixHopConv(3 * 60, 60, powers=[0, 1, 2])
        self.norm3 = BatchNorm(3 * 60)

        self.lin = Linear(3 * 60, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.7, training=self.training)

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.dropout(x, p=0.9, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.dropout(x, p=0.9, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.dropout(x, p=0.9, training=self.training)

        return self.lin(x)


model, data = MixHop().to(device), data.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40,
                                            gamma=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
