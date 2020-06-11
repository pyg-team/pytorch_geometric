import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PROTEINS')
dataset = TUDataset(path, name='PROTEINS').shuffle()
average_nodes = int(dataset.data.x.size(0) / len(dataset))
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DataLoader(test_dataset, batch_size=20)
val_loader = DataLoader(val_dataset, batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super(Net, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        num_nodes = ceil(0.5 * average_nodes)
        self.pool1 = Linear(hidden_channels, num_nodes)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = Linear(hidden_channels, num_nodes)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = F.relu(self.conv2(x, adj))
        s = self.pool2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), mc1 + mc2, o1 + o2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, mc_loss, o_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y.view(-1)) + mc_loss + o_loss
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred, mc_loss, o_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(pred, data.y.view(-1)) + mc_loss + o_loss
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

    return loss, correct / len(loader.dataset)


best_val_acc = test_acc = 0
best_val_loss = float('inf')
patience = start_patience = 50
for epoch in range(1, 15000):
    train_loss = train(epoch)
    _, train_acc = test(train_loader)
    val_loss, val_acc = test(val_loader)
    if val_loss < best_val_loss:
        test_loss, test_acc = test(test_loader)
        best_val_acc = val_acc
        patience = start_patience
    else:
        patience -= 1
        if patience == 0:
            break
    print('Epoch: {:03d}, '
          'Train Loss: {:.3f}, Train Acc: {:.3f}, '
          'Val Loss: {:.3f}, Val Acc: {:.3f}, '
          'Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, train_loss,
                                                       train_acc, val_loss,
                                                       val_acc, test_loss,
                                                       test_acc))
