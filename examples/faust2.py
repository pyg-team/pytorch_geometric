from __future__ import division, print_function

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import FAUST  # noqa
from torch_geometric.transform import CartesianAdj  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import (sparse_voxel_max_pool,
                                           dense_voxel_max_pool)  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'FAUST')
transform = CartesianAdj()
train_dataset = FAUST(path, train=True, transform=transform)
test_dataset = FAUST(path, train=False, transform=transform)
train_loader = DataLoader2(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=8)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 128, dim=3, kernel_size=5)
        self.conv4 = SplineConv(128, 256, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(8 * 256, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, data):
        data.input = F.elu(self.conv1(data.adj, data.input))
        data, _ = sparse_voxel_max_pool(data, 0.02, transform=transform)

        data.input = F.elu(self.conv2(data.adj, data.input))
        data, _ = sparse_voxel_max_pool(data, 0.04, transform=transform)

        data.input = F.elu(self.conv3(data.adj, data.input))
        data, _ = sparse_voxel_max_pool(data, 0.08, transform=transform)

        data.input = F.elu(self.conv4(data.adj, data.input))
        data, _ = dense_voxel_max_pool(data, 5, -5, 5)

        x = data.input.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        F.nll_loss(model(data), data.target).backward()
        optimizer.step()


def test(epoch):
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.cuda().to_variable('input')
        pred = model(data).data.max(1)[1]
        correct += pred.eq(data.target).sum()

    print('Epoch:', epoch, 'Test Accuracy:', correct / len(test_dataset))


for epoch in range(1, 101):
    train(epoch)
    test(epoch)
