from __future__ import division, print_function

import os.path as osp
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Compose
from torch_scatter import scatter_mean

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cuneiform2  # noqa
from torch_geometric.transform import (RandomRotate, RandomScale,
                                       RandomTranslate, CartesianAdj)  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa

transform = Compose([
    RandomRotate(0.6),
    RandomScale(1.4),
    RandomTranslate(0.1),
    CartesianAdj(),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Cuneiform')
train_dataset = Cuneiform2(path, transform=transform)
test_dataset = Cuneiform2(path, transform=CartesianAdj())

# Modify inputs.
# input = train_dataset.input
# input[:, :7] = 2 * input[:, :7] - 1
# input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], dim=1)
# train_dataset.input = input
# test_dataset.input = input

# Create random splits.
n = len(train_dataset)
step = (n + 10) // 10
perm = torch.randperm(n)
split = torch.arange(0, n + step, step).long()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(8, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(64, 30)

    def forward(self, data):
        x, adj = data.input, data.adj
        x = F.elu(self.conv1(adj, x))
        x = F.elu(self.conv2(adj, x))
        x = F.elu(self.conv3(adj, x))
        x = scatter_mean(Variable(data.batch.unsqueeze(1).expand_as(x)), x)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


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
        data = data.cuda().to_variable()

        optimizer.zero_grad()
        F.nll_loss(model(data), data.target).backward()
        optimizer.step()


def test(run, loader):
    model.eval()
    correct = 0
    num_examples = 0

    for data in loader:
        data = data.cuda().to_variable('input')
        pred = model(data).data.max(1)[1]
        correct += pred.eq(data.target).sum()
        num_examples += data.target.size(0)

    print('Run:', run, 'Test Accuracy:', correct / num_examples)
    return correct / num_examples


# 10-fold cross-validation.
acc = []
for i in range(split.size(0) - 1):
    print('Split {}:'.format(i + 1))

    if i == 0:
        train_dataset.split = perm[split[i + 1]:]
    elif i == split.size(0) - 2:
        train_dataset.split = perm[:split[i]]
    else:
        train_dataset.split = torch.cat([perm[:split[i]], perm[split[i + 1]:]])
    test_dataset.split = perm[split[i]:split[i + 1]]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    acc_split = []
    for run in range(1, 11):
        model.conv1.reset_parameters()
        model.conv2.reset_parameters()
        model.conv3.reset_parameters()
        model.fc1.reset_parameters()

        for epoch in range(1, 301):
            train(epoch)

        acc_split.append(test(run, test_loader))
    acc.append(acc_split)

print('Mean:', torch.Tensor(acc).mean(), 'Stddev:', torch.Tensor(acc).std())
