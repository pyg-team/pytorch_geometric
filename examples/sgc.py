import os.path as osp
import time

import torch
import torch.nn.functional as F

from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv

wall_clock_start = time.perf_counter()
seed_everything(123)

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SGConv(
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            K=2,
            cached=True,
        )

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model, data = Net().to(device), data.to(device)
model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=0.005)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


print(f'Total time before training begins took '
      f'{time.perf_counter() - wall_clock_start:.4f}s')
print('Training...')
times = []
best_val_acc = test_acc = 0
for epoch in range(1, 101):
    start = time.perf_counter()
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
    times.append(time.perf_counter() - start)

print(f'Average Epoch Time: {torch.tensor(times).mean():.4f}s')
print(f'Median Epoch Time: {torch.tensor(times).median():.4f}s')
print(f'Best Validation Accuracy: {100.0 * best_val_acc:.2f}%')
print(f'Test Accuracy: {100.0 * test_acc:.2f}%')
print(f'Total Program Runtime: '
      f'{time.perf_counter() - wall_clock_start:.4f}s')
