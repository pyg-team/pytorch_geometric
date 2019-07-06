import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import SyntheticDataset
from torch_geometric.transforms import OneHotDegree, HandleNodeAttention
from torch_geometric.transforms import Compose
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, SAGPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_scatter import scatter_mean

transform = Compose([HandleNodeAttention(), OneHotDegree(max_degree=14)])

train_path = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', 'TRIANGLES')
dataset = SyntheticDataset(train_path, name='TRIANGLES', use_node_attr=True,
                           transform=transform)

n_train, n_val, n_test_each = 30000, 5000, 5000

train_dataset = dataset[:n_train]
train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
val_loader = DataLoader(dataset[n_train:n_train + n_val], batch_size=60)
test_loader = DataLoader(dataset[n_train + n_val:], batch_size=60)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(train_dataset.num_features, 64), nn.ReLU(),
                nn.Linear(64, 64)))
        self.pool1 = SAGPooling(64, min_score=0.001, gnn='GCN')
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)))

        self.pool2 = SAGPooling(64, min_score=0.001, gnn='GCN')

        self.conv3 = GINConv(
            nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)))

        self.lin = torch.nn.Linear(64, 1)  # regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm, score = self.pool1(
            x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, perm, score = self.pool2(
            x, edge_index, None, batch)
        ratio = x.shape[0] / float(data.x.shape[0])

        x = F.relu(self.conv3(x, edge_index))
        x = gmp(x, batch)

        x = self.lin(x)

        # supervised node attention
        attn_loss_batch = scatter_mean(
            F.kl_div(
                torch.log(score + 1e-14), data.node_attention[perm],
                reduction='none'), batch)

        return x, attn_loss_batch, ratio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
print(model)
print('model size: %d trainable parameters' % np.sum(
    [np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, attn_loss, _ = model(data)

        loss = ((data.y - output.view_as(data.y))**2 + 100 * attn_loss).mean()

        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct, ratio_all = [], 0
    for data in loader:
        data = data.to(device)
        output, _, ratio = model(data)
        pred = output.round().long().view_as(data.y)
        correct += list(pred.eq(data.y.long()).data.cpu().numpy())
        ratio_all += ratio
    return np.array(correct), ratio_all / len(loader)


for epoch in range(1, 101):
    loss = train(epoch)
    train_correct, train_ratio = test(train_loader)
    val_correct, val_ratio = test(val_loader)
    test_correct, test_ratio = test(test_loader)

    train_acc = train_correct.sum() / len(train_correct)
    val_acc = val_correct.sum() / len(val_correct)

    # Test on two different subsets
    test_correct1 = test_correct[:n_test_each].sum()
    test_correct2 = test_correct[n_test_each:].sum()
    assert len(test_correct) == n_test_each * 2, len(test_correct)

    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.3f}, Val Acc: {:.3f}, '
          'Test Acc Orig: {:.3f} ({}/{}), '
          'Test Acc Large: {:.3f} ({}/{}), '
          'Train/Val/Test Pool Ratio={:.3f}/{:.3f}/{:.3f}'.format(
              epoch, loss, train_acc, val_acc, test_correct1 / n_test_each,
              test_correct1, n_test_each, test_correct2 / n_test_each,
              test_correct2, n_test_each, train_ratio, val_ratio, test_ratio))
