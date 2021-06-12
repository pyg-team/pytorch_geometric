import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, MemPooling, DeepGCNLayer
from torch import optim

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'PROTEINS_full')
dataset = TUDataset(path, name="PROTEINS_full", use_node_attr=True)
dataset.data.x = dataset.data.x[:, :-3]
dataset = dataset.shuffle()
n = (len(dataset)) // 10

test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]

test_loader = DataLoader(test_dataset, batch_size=20)
val_loader = DataLoader(val_dataset, batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32,
                 dropout=0.1):
        super(Net, self).__init__()

        self.dropout = dropout
        self.node_embed = nn.Linear(in_channels, hidden_channels)
        self.query_net = nn.ModuleList()
        for i in range(2):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout)
            norm = nn.BatchNorm1d(hidden_channels)
            act = nn.LeakyReLU()
            block = "res+"
            self.query_net.append(DeepGCNLayer(conv, norm, act, block,
                                               dropout))
        self.mem1 = MemPooling(hidden_channels, 80, 5, 10)
        self.mem2 = MemPooling(80, out_channels, 5, 1)

    def forward(self, x, edge_index, batch):
        x = self.node_embed(x)
        for lyrs in self.query_net:
            x = lyrs(x, edge_index)
        x, S = self.mem1(x, batch)
        x = F.dropout(x, p=self.dropout)
        x, S1 = self.mem2(F.leaky_relu(x))
        return F.log_softmax(
            F.leaky_relu(x).squeeze(1),
            dim=-1), MemPooling.kl_loss(S1) + MemPooling.kl_loss(S)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-5)


def train(model, optimizer, loader):
    model.train()
    kl_loss = 0.0
    loss = 0.0

    model.mem1.k.requires_grad = False
    model.mem2.k.requires_grad = False
    for d in loader:
        d = d.to(device)
        out, _ = model(d.x, d.edge_index, d.batch)
        loss = F.nll_loss(out, d.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.mem1.k.requires_grad = True
    model.mem2.k.requires_grad = True
    for d in loader:
        d = d.to(device)
        _, kl = model(d.x, d.edge_index, d.batch)
        kl_loss += kl
    (kl_loss / len(loader.dataset)).backward()
    optimizer.step()
    optimizer.zero_grad()


def evaluate(model, loader):
    model.eval()
    accu = 0.0
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            out, kl = model(d.x, d.edge_index, d.batch)
            accu += d.y.eq(out.max(1)[1]).sum()

    return accu / len(loader.dataset)


epochs = 2000
patience = start_patience = 250
test_acc = best_val_acc = 0.0

for e in range(epochs):
    train(model, optimizer, train_loader)
    val_acc = evaluate(model, val_loader)
    if (e + 1) % 500 == 0:
        optimizer.param_groups[0]['lr'] *= 0.5
    if best_val_acc < val_acc:
        patience = start_patience
        best_val_acc = val_acc
        test_acc = evaluate(model, test_loader)
    else:
        patience -= 1
    if patience <= 0:
        print('Early stopping')
        break
    print(
        'Epoch {}: Validation accuracy = {:.3f}, Test accuracy= {:.3f}'.format(
            e, val_acc, test_acc))
