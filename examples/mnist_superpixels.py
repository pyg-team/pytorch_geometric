from __future__ import division, print_function

import os
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixels  # noqa
from torch_geometric.transforms import CartesianAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import max_pool  # noqa
from torch_geometric.sparse import SparseTensor  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'MNISTSuperpixels')
train_dataset = MNISTSuperpixels(path, train=True, transform=CartesianAdj())
test_dataset = MNISTSuperpixels(path, train=False, transform=CartesianAdj())

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def to_cartesian_adj(index, position):
    row, col = index
    n, dim = position.size()
    cartesian = position[col] - position[row]
    cartesian *= 1 / (2 * cartesian.abs().max())
    cartesian += 0.5
    return SparseTensor(index, cartesian, torch.Size([n, n, dim]))


def max_grid_pool(x, position, adj, batch, size):
    cell_size = position.new(2).fill_(size)
    cluster, batch = grid_cluster(position, cell_size, batch)
    x, index, position = max_pool(x, adj._indices(), position, cluster)
    adj = to_cartesian_adj(index, position)
    return x, position, adj, batch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x, position, adj, batch):
        x = F.elu(self.conv1(adj, x))
        # x, position, adj, batch = max_grid_pool(x, position, adj, batch, 3)
        x = F.elu(self.conv2(adj, x))
        # x, position, adj, batch = max_grid_pool(x, position, adj, batch, 5)
        x = F.elu(self.conv3(adj, x))

        # Batch average.
        t_scatter_mean = time.process_time()
        batch = Variable(batch.view(-1, 1).expand(batch.size(0), 64))
        x = scatter_mean(batch, x)
        t_scatter_mean = time.process_time() - t_scatter_mean
        print('t_scatter_mean',t_scatter_mean)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    t_all = 0
    for i, data in enumerate(train_loader):
        t_all = time.process_time() - t_all
        print('t_all', t_all)
        t_all = time.process_time()
        input, position = data['input'].view(-1, 1), data['position']
        adj, target = data['adj']['content'], data['target']
        batch_len = data['adj']['slice'].size(0)
        batch = torch.arange(0, batch_len).view(-1, 1)
        batch = batch.repeat(1, 75).view(-1).long()

        if torch.cuda.is_available():
            input, position = input.cuda(), position.cuda()
            adj, target = adj.cuda(), target.cuda()
            batch = batch.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        t_forward = time.process_time()
        output = model(input, position, adj, batch)
        t_forward = time.process_time() - t_forward
        loss = F.nll_loss(output, target)
        t_backward = time.process_time()
        loss.backward()
        t_backward = time.process_time() - t_backward
        optimizer.step()
        print('forward', t_forward, 'backward', t_backward)
        print(i, loss.data[0])


def test(epoch):
    model.eval()
    correct = 0

    for i, data in enumerate(test_loader):
        input, position = data['input'].view(-1, 1), data['position']
        adj, target = data['adj']['content'], data['target']
        batch_len = data['adj']['slice'].size(0)
        batch = torch.arange(0, batch_len).view(-1, 1)
        batch = batch.repeat(1, 75).view(-1).long()

        if torch.cuda.is_available():
            input, position = input.cuda(), position.cuda()
            adj, target = adj.cuda(), target.cuda()
            batch = batch.cuda()

        output = model(Variable(input), position, adj, batch)
        pred = output.data.max(1)[1]
        correct += pred.eq(target).cpu().sum()

    print('Epoch:', epoch, 'Test Accuracy:', correct / len(test_dataset))


for epoch in range(1, 31):
    train(epoch)
    test(epoch)
