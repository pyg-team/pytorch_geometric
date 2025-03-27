import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SuperGATConv

wall_clock_start = time.perf_counter()
seed_everything(123)

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = SuperGATConv(dataset.num_features, 8, heads=8,
                                  dropout=0.6, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)
        self.conv2 = SuperGATConv(8 * 8, dataset.num_classes, heads=8,
                                  concat=False, dropout=0.6,
                                  attention_type='MX', edge_sample_ratio=0.8,
                                  is_undirected=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        att_loss += self.conv2.get_attention_loss()
        return F.log_softmax(x, dim=-1), att_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out, att_loss = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss += 4.0 * att_loss
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index)[0], []
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
for epoch in range(1, 501):
    start = time.perf_counter()
    train(data)
    train_acc, val_acc, tmp_test_acc = test(data)
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
