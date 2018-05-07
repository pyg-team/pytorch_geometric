import os
import sys
import random

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from torch.autograd import Variable
from torch_scatter import scatter_mean

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet10  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.transform import (NormalizeScale, LogCartesian,
                                       RandomTranslate)  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch.nn import Linear as Lin  # noqa
from torch_geometric.nn.functional import sparse_voxel_max_pool  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10')

transform = LogCartesian(30)
init_transform = Compose([NormalizeScale(), RandomTranslate(0.001)])
train_dataset = ModelNet10(path, True, init_transform)
test_dataset = ModelNet10(path, False, init_transform)
batch_size = 6
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv12 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv2 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.conv22 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.conv32 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.conv42 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv5 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.conv52 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(5 * 64, 10)

        self.att1 = Lin(32, 2)
        self.att2 = Lin(64, 2)
        self.att3 = Lin(64, 2)
        self.att4 = Lin(64, 2)

    def pool_args(self, mean, x):
        if not self.training:
            return 1 / mean, 0
        size = [1 / random.uniform(mean - x, mean + x) for _ in range(3)]
        # size = 1 / mean
        start = [random.uniform(-1 / (mean - x), 0) for _ in range(3)]
        # start = 0
        return size, start

    def forward(self, data):
        f1 = F.relu(self.conv1(data.adj, data.input))
        f1 = F.relu(self.conv12(data.adj, f1))
        batch = Variable(data.batch.view(-1, 1).expand(data.batch.size(0), 64))
        f1_res = scatter_mean(batch, f1)

        size, start = self.pool_args(32, 8)
        data2, _ = sparse_voxel_max_pool(
            data, size, start, transform, weight=f1[:, 0])

        f2 = F.relu(self.conv2(data2.adj, data2.input))
        f2 = F.relu(self.conv22(data2.adj, f2))
        batch = Variable(
            data2.batch.view(-1, 1).expand(data2.batch.size(0), 64))
        f2_res = scatter_mean(batch, f2)

        size, start = self.pool_args(16, 4)
        data3, _ = sparse_voxel_max_pool(
            data2, size, start, transform, weight=f2[:, 0])

        f3 = F.relu(self.conv3(data3.adj, data3.input))
        f3 = F.relu(self.conv32(data3.adj, f3))
        batch = Variable(
            data3.batch.view(-1, 1).expand(data3.batch.size(0), 64))
        f3_res = scatter_mean(batch, f3)

        size, start = self.pool_args(8, 2)
        data4, _ = sparse_voxel_max_pool(
            data3, size, start, transform, weight=f3[:, 0])

        f4 = F.relu(self.conv4(data4.adj, data4.input))
        f4 = F.relu(self.conv42(data4.adj, f4))
        batch = Variable(
            data4.batch.view(-1, 1).expand(data4.batch.size(0), 64))
        f4_res = scatter_mean(batch, f4)

        size, start = self.pool_args(4, 1)
        data5, _ = sparse_voxel_max_pool(
            data4, size, start, transform, weight=f4[:, 0])

        f5 = F.relu(self.conv5(data5.adj, data5.input))
        f5 = F.relu(self.conv52(data5.adj, f5))
        batch = Variable(
            data5.batch.view(-1, 1).expand(data5.batch.size(0), 64))
        f5_res = scatter_mean(batch, f5)

        x = torch.cat([f1_res, f2_res, f3_res, f4_res, f5_res], dim=1)

        # data.input = F.relu(self.conv5(data.adj, data.input))
        # data, _ = dense_voxel_max_pool(data, 1, -0.5, 1.5)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    if epoch == 11:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    if epoch == 41:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00005

    if epoch == 46:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    for data in train_loader:
        data = data.cuda().to_variable(['input', 'target', 'pos'])
        data = transform(data)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.target)
        loss.backward()
        optimizer.step()


def test(epoch, loader, dataset, str):
    model.eval()
    correct = 0

    for data in loader:
        data = data.cuda().to_variable(['input', 'pos'])
        data = transform(data)
        pred = model(data).data.max(1)[1]
        correct += pred.eq(data.target).cpu().sum()

    print('Epoch:', epoch, str, 'Accuracy:', correct / len(dataset))


for epoch in range(1, 51):
    train(epoch)
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, test_loader, test_dataset, 'Test')
