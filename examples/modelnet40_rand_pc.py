import os
import sys
import random

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet40RandPC  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import (NormalizeScale, RandomFlip,
                                       CartesianAdj, RandomTranslate)  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import (sparse_voxel_max_pool,
                                           dense_voxel_max_pool)  # noqa
from torch_geometric.visualization.model import show_model  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet40RandPC')

transform = CartesianAdj()
train_transform = Compose([NormalizeScale(),
                           RandomTranslate(0.001),
                           transform])
test_transform = Compose([NormalizeScale(), transform])
train_dataset = ModelNet40RandPC(path, True, transform=train_transform)
test_dataset = ModelNet40RandPC(path, False, transform=test_transform)
batch_size = 6
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.conv2 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(64, 96, dim=3, kernel_size=5)
        self.conv5 = SplineConv(96, 96, dim=3, kernel_size=5)
        self.conv6 = SplineConv(96, 128, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(8 * 128, 10)

    def pool_args(self, mean, x):
        #return 1 / mean, 0
        if not self.training:
            return 1 / mean, 0
        size = 1 / random.uniform(mean - x, mean + x)
        start = random.uniform(-1 / (mean - x), 0)
        return size, start

    def forward(self, data):
        data.input = F.elu(self.conv1(data.adj, data.input))
        size, start = self.pool_args(64, 16)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input = F.elu(self.conv2(data.adj, data.input))
        size, start = self.pool_args(32, 8)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input = F.elu(self.conv3(data.adj, data.input))
        size, start = self.pool_args(16, 4)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input = F.elu(self.conv4(data.adj, data.input))
        size, start = self.pool_args(8, 2)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input = F.elu(self.conv5(data.adj, data.input))
        size, start = self.pool_args(4, 1)
        data, _ = sparse_voxel_max_pool(data, size, start, transform)

        data.input = F.elu(self.conv6(data.adj, data.input))
        data, _ = dense_voxel_max_pool(data, 1, -0.5, 1.5, transform)

        x = data.input.view(-1, 8 * 128)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    if epoch == 21:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    if epoch == 41:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    #if epoch == 61:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.000001

    #if epoch == 81:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.000001

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.target)
        loss.backward()
        optimizer.step()


def test(epoch, loader, dataset, str):
    model.eval()
    correct = 0

    for data in loader:
        data = data.cuda().to_variable(['input'])
        pred = model(data).data.max(1)[1]
        correct += pred.eq(data.target).cpu().sum()

    print('Epoch:', epoch, str, 'Accuracy:', correct / len(dataset))


for epoch in range(1, 101):
    train()
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, test_loader, test_dataset, 'Test')
