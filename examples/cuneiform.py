from __future__ import division, print_function

import os
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cuneiform  # noqa
from torch_geometric.transforms import (RandomRotate, RandomScale,
                                        RandomTranslate, CartesianAdj)  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import batch_average  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'Cuneiform')
n = 267
perm = torch.randperm(n)
perm = torch.load('/tmp/perm.pt')
split = torch.arange(0, n + 27, 27, out=torch.LongTensor())
train_transform = Compose([
    RandomRotate(0.6),
    RandomScale(1.4),
    RandomTranslate(0.1),
    CartesianAdj(),
])
test_transform = CartesianAdj()
train_dataset = Cuneiform(path, split=None, transform=train_transform)
test_dataset = Cuneiform(path, split=None, transform=test_transform)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(8, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 124, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(124, 30)

    def forward(self, x, adj, slice):
        x = F.elu(self.conv1(adj, x))
        x = F.elu(self.conv2(adj, x))
        x = F.elu(self.conv3(adj, x))
        x = batch_average(x, slice)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01

    if epoch == 201:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for data in train_loader:
        adj, slice = data['adj']['content'], data['adj']['slice'][:, 0]
        input, target = data['input'], data['target']
        input *= 2
        input -= 1
        input = torch.cat([input, input.new(input.size(0)).fill_(1)], dim=1)

        if torch.cuda.is_available():
            adj, slice = adj.cuda(), slice.cuda()
            input, target = input.cuda(), target.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model(input, adj, slice)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(epoch, loader, string):
    model.eval()

    correct = 0
    num_examples = 0

    for data in loader:
        adj, slice = data['adj']['content'], data['adj']['slice'][:, 0]
        input, target = data['input'], data['target']
        input *= 2
        input -= 1
        input = torch.cat([input, input.new(input.size(0)).fill_(1)], dim=1)
        num_examples += target.size(0)

        if torch.cuda.is_available():
            adj, slice = adj.cuda(), slice.cuda()
            input, target = input.cuda(), target.cuda()

        output = model(Variable(input), adj, slice)
        pred = output.data.max(1)[1]
        correct += pred.eq(target).cpu().sum()

    acc = correct / num_examples
    print('Epoch', epoch, string, acc)
    return acc


# 10-fold cross-validation.
accs = []
for i in range(split.size(0) - 1):
    print('Split', i)

    test_split = perm[split[i]:split[i + 1]]

    if i == 0:
        train_split = perm[split[i + 1]:]
    elif i == split.size(0) - 2:
        train_split = perm[:split[i]]
    else:
        train_split = torch.cat([perm[:split[i]], perm[split[i + 1]:]])

    train_dataset.split = train_split
    test_dataset.split = test_split

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    accs_single = []
    for _ in range(10):
        model.conv1.reset_parameters()
        model.conv2.reset_parameters()
        model.fc1.reset_parameters()
        for epoch in range(1, 301):
            train(epoch)
        acc = test(epoch, test_loader, ' Test Accuracy')
        accs_single.append(acc)
    accs.append(accs_single)

acc = torch.FloatTensor(accs)
torch.save(acc, '/tmp/acc.pt')
print('Mean:', acc.mean(), 'Stddev:', acc.std())
