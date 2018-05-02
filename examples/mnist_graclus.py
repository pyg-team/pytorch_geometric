from __future__ import division, print_function

import os.path as osp

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import DataLoader, normalized_cut
from torch_geometric.transform import CartesianAdj
from torch_geometric.nn.modules import SplineConv
from torch_geometric.nn.functional.pool import graclus_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
train_dataset = MNISTSuperpixels(path, True, transform=CartesianAdj())
test_dataset = MNISTSuperpixels(path, False, transform=CartesianAdj())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, pos.size(0))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, data):
        data.input = F.elu(self.conv1(data.input, data.index, data.weight))
        weight = normalized_cut_2d(data.index, data.pos)
        data = graclus_pool(data, weight, transform=CartesianAdj())
        data.weight = Variable(data.weight)

        data.input = F.elu(self.conv2(data.input, data.index, data.weight))
        weight = normalized_cut_2d(data.index, data.pos)
        data = graclus_pool(data, weight, transform=CartesianAdj())

        x = data.input
        index = Variable(data.batch.unsqueeze(1).expand(x.size()))
        x = scatter_mean(index, x, dim=0)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        F.nll_loss(model(data), data.target).backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.cuda().to_variable()
        pred = model(data).max(1)[1]
        correct += pred.eq(data.target).sum().item()
    return correct / len(test_dataset)


for epoch in range(1, 31):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
