import os
import sys
import random

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet10  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import (NormalizeScale, RandomFlip,
                                       LogCartesianAdj)  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import (sparse_voxel_max_pool as max_pool,
                                           dense_voxel_max_pool)  # noqa
from torch_geometric.visualization.model import show_model  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10')

transform = LogCartesianAdj(30)
train_transform = Compose([
    NormalizeScale(),
    RandomFlip(0),
    # RandomFlip(1),
    transform,
])
test_transform = Compose([NormalizeScale(), transform])
train_dataset = ModelNet10(path, True, transform=train_transform)
test_dataset = ModelNet10(path, False, transform=test_transform)
batch_size = 6
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size)


# To beat: 95.40 %
class Net(nn.Module):
    def pool_args(self, min, max):
        if not self.training:
            return 1 / (0.5 * min + max), 0
        return 1 / random.uniform(min, max), random.uniform(-1 / min, 0)

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 128, dim=3, kernel_size=5)
        self.conv4 = SplineConv(128, 256, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(8 * 256, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, data):
        size, start = self.pool_args(32, 64)
        data = max_pool(data, size, start, transform)

        data.input = F.elu(self.conv1(data.adj, data.input))
        size, start = self.pool_args(16, 32)
        data = max_pool(data, size, start, transform)

        data.input = F.elu(self.conv2(data.adj, data.input))
        size, start = self.pool_args(8, 16)
        data = max_pool(data, size, start, transform)

        data.input = F.elu(self.conv3(data.adj, data.input))
        size, start = self.pool_args(4, 8)
        data = max_pool(data, size, start, transform)

        data.input = F.elu(self.conv4(data.adj, data.input))
        data = dense_voxel_max_pool(data, 0.5, 0, 1, transform)

        x = data.input.view(-1, 8 * 256)
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

    if epoch == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.target)
        loss.backward()
        optimizer.step()


def test(epoch, loader, dataset, str):
    model.eval()
    correct = 0

    for data in loader:
        data = data.cuda().to_variable(['input'])
        output = model(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(data.target).cpu().sum()

    print('Epoch:', epoch, str, 'Accuracy:', correct / len(dataset))


for epoch in range(1, 51):
    train(epoch)
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, test_loader, test_dataset, 'Test')
