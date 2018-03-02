import os
import sys
import random
import time

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet10PC  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import NormalizeScale, CartesianAdj  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import (sparse_voxel_max_pool,
                                           dense_voxel_max_pool)  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10PC')

train_dataset = ModelNet10PC(path, True, transform=NormalizeScale())
test_dataset = ModelNet10PC(path, False, transform=NormalizeScale())
train_loader = DataLoader2(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=32)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.conv2 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 128, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(8 * 128, 256)
        self.fc2 = nn.Linear(256, 10)

    def pool_args(self, mean, x):
        if not self.training:
            return 1 / mean, 0
        size = 1 / random.uniform(mean - x, mean + x)
        start = random.uniform(-1 / (mean - x), 0)
        return size, start

    def forward(self, data):
        # data.input = F.elu(self.conv1(data.adj, data.input))
        size, start = self.pool_args(13, 5)
        data, _ = sparse_voxel_max_pool(data, size, start, CartesianAdj())

        data.input = F.elu(self.conv1(data.adj, data.input))
        size, start = self.pool_args(7, 3)
        data, _ = sparse_voxel_max_pool(data, size, start, CartesianAdj())

        data.input = F.elu(self.conv2(data.adj, data.input))
        size, start = self.pool_args(5, 2)
        data, _ = sparse_voxel_max_pool(data, size, start, CartesianAdj())

        data.input = F.elu(self.conv3(data.adj, data.input))
        data, _ = dense_voxel_max_pool(data, 1 / 2, 0, 1)

        x = data.input.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    if epoch == 8:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        t_forward = time.perf_counter()
        output = model(data)
        loss = F.nll_loss(output, data.target)
        torch.cuda.synchronize()
        t_forward = time.perf_counter() - t_forward
        t_backward = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_backward = time.perf_counter() - t_backward
        optimizer.step()


def test(epoch, loader, dataset, str):
    model.eval()
    correct = 0

    for data in loader:
        data = data.cuda().to_variable(['input'])
        pred = model(data).data.max(1)[1]
        correct += pred.eq(data.target).cpu().sum()

    print('Epoch:', epoch, str, 'Accuracy:', correct / len(dataset))


for epoch in range(1, 21):
    train(epoch)
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, test_loader, test_dataset, 'Test')
