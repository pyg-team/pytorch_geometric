import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet10  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import (NormalizeScale, LogCartesianAdj)  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import (sparse_voxel_max_pool,
                                           dense_voxel_max_pool)  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10')

transform = LogCartesianAdj(30)
init_transform = Compose([NormalizeScale(), transform])
train_dataset = ModelNet10(path, True, transform=init_transform)
test_dataset = ModelNet10(path, False, transform=init_transform)
batch_size = 8
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 32, dim=3, kernel_size=5)
        self.conv3 = SplineConv(32, 32, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(8 * 32, 10)

    def forward(self, data):
        data, _ = sparse_voxel_max_pool(data, 1 / 16, 0, transform)
        data.input = F.elu(self.conv1(data.adj, data.input))

        data, _ = sparse_voxel_max_pool(data, 1 / 8, 0, transform)
        data.input = F.elu(self.conv2(data.adj, data.input))

        data, _ = sparse_voxel_max_pool(data, 1 / 4, 0, transform)
        # data.input = F.dropout(data.input, training=self.training)
        data.input = F.elu(self.conv3(data.adj, data.input))

        data, _ = dense_voxel_max_pool(data, 0.5, 0, 1)

        x = data.input.view(-1, self.fc1.weight.size(1))
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

    if epoch == 21:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

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
        data = data.cuda().to_variable('input')
        pred = model(data).data.max(1)[1]
        correct += pred.eq(data.target).cpu().sum()

    print('Epoch:', epoch, str, 'Accuracy:', correct / len(dataset))


for epoch in range(1, 31):
    train(epoch)
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, test_loader, test_dataset, 'Test')
