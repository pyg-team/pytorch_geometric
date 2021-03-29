import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import torch_geometric.transforms as T
from torch_geometric.nn.models import CorrectAndSmooth

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OGB')
dataset = PygNodePropPredDataset('ogbn-products', root,
                                 transform=T.ToSparseTensor())
evaluator = Evaluator(name='ogbn-products')
split_idx = dataset.get_idx_split()
data = dataset[0]


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.lins = ModuleList([Linear(in_channels, hidden_channels)])
        self.bns = ModuleList([BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

    def forward(self, x):
        for lin, bn in zip(self.lins[:-1], self.bns):
            x = bn(lin(x).relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(dataset.num_features, dataset.num_classes, hidden_channels=200,
            num_layers=3, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

x, y = data.x.to(device), data.y.to(device)
train_idx = split_idx['train'].to(device)
val_idx = split_idx['valid'].to(device)
test_idx = split_idx['test'].to(device)
x_train, y_train = x[train_idx], y[train_idx]


def train():
    model.train()
    optimizer.zero_grad()
    out = model(x_train)
    loss = criterion(out, y_train.view(-1))
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(out=None):
    model.eval()
    out = model(x) if out is None else out
    pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': y[train_idx],
        'y_pred': pred[train_idx]
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y[val_idx],
        'y_pred': pred[val_idx]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[test_idx],
        'y_pred': pred[test_idx]
    })['acc']
    return train_acc, val_acc, test_acc, out


best_val_acc = 0
for epoch in range(1, 301):
    loss = train()
    train_acc, val_acc, test_acc, out = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        y_soft = out.softmax(dim=-1)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

adj_t = data.adj_t.to(device)
deg = adj_t.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow_(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t

post = CorrectAndSmooth(num_correction_layers=50, correction_alpha=1.0,
                        num_smoothing_layers=50, smoothing_alpha=0.8,
                        autoscale=False, scale=20.)

print('Correct and smooth...')
y_soft = post.correct(y_soft, y_train, train_idx, DAD)
y_soft = post.smooth(y_soft, y_train, train_idx, DA)
print('Done!')
train_acc, val_acc, test_acc, _ = test(y_soft)
print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
