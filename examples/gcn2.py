import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import GCN2Conv, BatchNorm
import torch_geometric.transforms as T

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=path,
                                 transform=T.ToSparseTensor())
split_idx = dataset.get_idx_split()
evaluator = Evaluator('ogbn-arxiv')

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 alpha, dropout=0.):

        super(Net, self).__init__()

        self.lin1 = Linear(in_channels, hidden_channels)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, cached=True))
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.lin2 = Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = F.relu(self.lin1(x))

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.dropout(x, self.dropout, training=self.training)
            x = batch_norm(x)
            x = F.relu(conv(x, x_0, adj_t)) + x

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, 256, dataset.num_classes, num_layers=8,
            alpha=0.5, dropout=0.1).to(device)
data = data.to(device)
train_idx = split_idx['train'].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(pred, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()

    pred = model(data.x, data.adj_t).argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


for epoch in range(1, 1001):
    loss = train()
    train_acc, val_acc, test_acc = test()

    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
