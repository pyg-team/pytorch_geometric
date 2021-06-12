from torch_geometric.datasets import TUDataset

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, MemPooling
from torch import optim

dataset = TUDataset(root='data', name="PROTEINS")
dataset = dataset.shuffle()
average_nodes = dataset.data.x.shape[0] // len(dataset)
n = (len(dataset) + 9) // 10

test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]

test_loader = DataLoader(test_dataset, batch_size=20)
val_loader = DataLoader(val_dataset, batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64,
                 dropout=0.1):
        super(Net, self).__init__()

        self.dropout = dropout
        self.out_channels = out_channels
        self.query_net = nn.ModuleList([
            GATConv(in_channels, hidden_channels, 3, dropout=dropout),
            GATConv(hidden_channels * 3, hidden_channels, 3, dropout=dropout),
            GATConv(hidden_channels * 3, hidden_channels, 3, dropout=dropout,
                    concat=False)
        ])

        self.mem1 = MemPooling(hidden_channels, 80, 5, 10)
        self.mem2 = MemPooling(80, out_channels, 5, 1)

    def forward(self, x, edge_index, batch):
        for lyrs in self.query_net:
            x = F.leaky_relu(lyrs(x, edge_index))
        x, S = self.mem1(x, batch)
        x = F.dropout(x, p=self.dropout)
        x, S1 = self.mem2(F.leaky_relu(x))
        return F.log_softmax(
            F.leaky_relu(x).squeeze(1),
            dim=-1), MemPooling.kl_loss(S1) + MemPooling.kl_loss(S)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(3, 2).to(device)
opt = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0)


def train(model, optim, loader):
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
        optim.step()
        optim.zero_grad()

    model.mem1.k.requires_grad = True
    model.mem2.k.requires_grad = True
    for d in loader:
        d = d.to(device)
        _, kl = model(d.x, d.edge_index, d.batch)
        kl_loss += kl
    (kl_loss / len(loader.dataset)).backward()
    optim.step()
    optim.zero_grad()


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
patience = 200
curr_patience = 200
best_accu = 0.0
for e in range(epochs):
    train(model, opt, train_loader)
    val_accu = evaluate(model, val_loader)
    if best_accu < val_accu:
        curr_patience = patience
        best_accu = val_accu
    else:
        curr_patience -= 1

    if curr_patience <= 0:
        print('Early stopping')
        break
    print(f'Epoch {e}: Validation accuracy = {val_accu}')

print(f'Test accuracy= {evaluate(model, test_loader)}')
