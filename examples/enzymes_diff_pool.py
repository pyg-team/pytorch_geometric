import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

max_nodes = 100


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


class MyTransform(object):
    def __call__(self, data):
        # Only use node attributes.
        data.x = data.x[:, :-3]

        # Add self loops.
        arange = torch.arange(data.adj.size(-1), dtype=torch.long)
        data.adj[arange, arange] = 1

        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES')
dataset = TUDataset(
    path,
    name='ENZYMES',
    transform=T.Compose([T.ToDense(max_nodes),
                         MyTransform()]),
    pre_filter=MyFilter())
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
train_loader = DenseDataLoader(train_dataset, batch_size=20)


class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 lin=True,
                 norm=True,
                 norm_embed=True):
        super(GNN, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, norm,
                                   norm_embed)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, norm,
                                   norm_embed)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, norm,
                                   norm_embed)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.gnn1_pool = GNN(18, 64, int(0.1 * max_nodes), norm=False)
        self.gnn1_embed = GNN(18, 64, 64, lin=False, norm=False)

        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False, norm=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 6)

    def forward(self, data):
        x, adj = data.x, data.adj

        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        x, adj, reg1 = dense_diff_pool(x, adj, s, data.mask)
        x = self.gnn2_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), reg1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, reg = model(data)
        loss = F.nll_loss(output, data.y.view(-1)) + reg
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)[0].max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


best_val_acc = test_acc = 0
for epoch in range(1, 1001):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                     val_acc, test_acc))
