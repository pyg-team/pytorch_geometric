import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv, SAGPooling
from torch_geometric.nn import global_max_pool
from torch_scatter import scatter_mean


class HandleNodeAttention(object):
    def __call__(self, data):
        data.attn = torch.softmax(data.x, dim=0).flatten()
        data.x = None
        return data


transform = T.Compose([HandleNodeAttention(), T.OneHotDegree(max_degree=14)])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TRIANGLES')
dataset = TUDataset(path, name='TRIANGLES', use_node_attr=True,
                    transform=transform)

train_loader = DataLoader(dataset[:30000], batch_size=60, shuffle=True)
val_loader = DataLoader(dataset[30000:35000], batch_size=60)
test_loader = DataLoader(dataset[35000:], batch_size=60)


class Net(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = GINConv(Seq(Lin(in_channels, 64), ReLU(), Lin(64, 64)))
        self.pool1 = SAGPooling(64, min_score=0.001, GNN=GCNConv)
        self.conv2 = GINConv(Seq(Lin(64, 64), ReLU(), Lin(64, 64)))
        self.pool2 = SAGPooling(64, min_score=0.001, GNN=GCNConv)
        self.conv3 = GINConv(Seq(Lin(64, 64), ReLU(), Lin(64, 64)))

        self.lin = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm, score = self.pool1(
            x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, perm, score = self.pool2(
            x, edge_index, None, batch)
        ratio = x.size(0) / data.x.size(0)

        x = F.relu(self.conv3(x, edge_index))
        x = global_max_pool(x, batch)
        x = self.lin(x).view(-1)

        attn_loss = F.kl_div(torch.log(score + 1e-14), data.attn[perm],
                             reduction='none')
        attn_loss = scatter_mean(attn_loss, batch)

        return x, attn_loss, ratio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, attn_loss, _ = model(data)
        loss = ((out - data.y).pow(2) + 100 * attn_loss).mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    return total_loss / len(train_loader.dataset)


def test(loader):
    model.eval()

    corrects, total_ratio = [], 0
    for data in loader:
        data = data.to(device)
        out, _, ratio = model(data)
        pred = out.round().to(torch.long)
        corrects.append(pred.eq(data.y.to(torch.long)))
        total_ratio += ratio
    return torch.cat(corrects, dim=0), total_ratio / len(loader)


for epoch in range(1, 301):
    loss = train(epoch)
    train_correct, train_ratio = test(train_loader)
    val_correct, val_ratio = test(val_loader)
    test_correct, test_ratio = test(test_loader)

    train_acc = train_correct.sum().item() / train_correct.size(0)
    val_acc = val_correct.sum().item() / val_correct.size(0)

    test_acc1 = test_correct[:5000].sum().item() / 5000
    test_acc2 = test_correct[5000:].sum().item() / 5000

    print(('Epoch: {:03d}, Loss: {:.4f}, Train: {:.3f}, Val: {:.3f}, '
           'Test Orig: {:.3f}, Test Large: {:.3f}, '
           'Train/Val/Test Ratio={:.3f}/{:.3f}/{:.3f}').format(
               epoch, loss, train_acc, val_acc, test_acc1, test_acc2,
               train_ratio, val_ratio, test_ratio))
