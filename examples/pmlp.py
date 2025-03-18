import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import PMLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wall_clock_start = time.perf_counter()
seed_everything(123)

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

model = PMLP(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
    num_layers=2,
    dropout=0.5,
    norm=False,
).to(device)

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x)  # MLP during training.
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


print(f'Total time before training begins took '
      f'{time.perf_counter() - wall_clock_start:.4f}s')
print('Training...')
times = []
best_val_acc = final_test_acc = 0
for epoch in range(1, 201):
    start = time.perf_counter()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
    times.append(time.perf_counter() - start)

print(f'Average Epoch Time: {torch.tensor(times).mean():.4f}s')
print(f'Median Epoch Time: {torch.tensor(times).median():.4f}s')
print(f'Best Validation Accuracy: {100.0 * best_val_acc:.2f}%')
print(f'Test Accuracy: {100.0 * test_acc:.2f}%')
print(f'Total Program Runtime: '
      f'{time.perf_counter() - wall_clock_start:.4f}s')
