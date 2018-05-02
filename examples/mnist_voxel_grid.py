from __future__ import division, print_function

import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import DataLoader
from torch_geometric.transform import CartesianAdj
from torch_geometric.nn.modules import SplineConv
from torch_geometric.nn.functional.pool import sparse_voxel_grid_pool
from torch_geometric.nn.functional.pool import dense_voxel_grid_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
train_dataset = MNISTSuperpixels(path, True, transform=CartesianAdj())
test_dataset = MNISTSuperpixels(path, False, transform=CartesianAdj())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(4 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, data):
        data.input = F.elu(self.conv1(data.input, data.index, data.weight))
        data = sparse_voxel_grid_pool(data, 5, 0, 28, CartesianAdj())

        data.input = F.elu(self.conv2(data.input, data.index, data.weight))
        data = sparse_voxel_grid_pool(data, 7, 0, 28, CartesianAdj())

        data.input = F.elu(self.conv3(data.input, data.index, data.weight))
        x = dense_voxel_grid_pool(data, 14, 0, 27.99)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 16:
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


for epoch in range(1, 21):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
