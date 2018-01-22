from __future__ import division, print_function

import os
import sys

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
from torch_geometric.datasets.utils.tu_format import read_file

mode=1
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'Cuneiform_{}'.format(mode))

train_transform = Compose([
    RandomRotate(0.6),
    RandomScale(1.4),
    RandomTranslate(0.1),
    CartesianAdj(),
])
test_transform = CartesianAdj()
train_dataset = Cuneiform(path, mode=mode, transform=train_transform)
test_dataset = Cuneiform(path, mode=mode, transform=test_transform)

# Modify inputs.
input = train_dataset.input
input[:, :7] = 2 * input[:, :7] - 1
input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], dim=1)
train_dataset.input = input
test_dataset.input = input

set = read_file(train_dataset.raw_folder, train_dataset.prefix, 'graph_sets')
set = set.byte()
train_dataset.split = train_dataset.split[set == 0]
test_dataset.split = test_dataset.split[set]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(9, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 128, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(128, 30)

    def forward(self, x, adj, slice):
        x = F.elu(self.conv1(adj, x))
        x = F.elu(self.conv2(adj, x))
        x = F.elu(self.conv3(adj, x))
        x = batch_average(x, slice)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return F.softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 501:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for data in train_loader:
        adj, slice = data['adj']['content'], data['adj']['slice'][:, 0]
        input, target = data['input'], data['target']

        if torch.cuda.is_available():
            adj, slice = adj.cuda(), slice.cuda()
            input, target = input.cuda(), target.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model(input, adj, slice)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(clx):
    model.eval()

    conf = torch.FloatTensor([])
    label = torch.LongTensor([])

    for data in test_loader:
        adj, slice = data['adj']['content'], data['adj']['slice'][:, 0]
        input, target = data['input'], data['target']

        if torch.cuda.is_available():
            adj, slice = adj.cuda(), slice.cuda()
            input = input.cuda()

        output = model(Variable(input), adj, slice)
        output = output.data.cpu()
        output = output[:, clx]
        conf = torch.cat([conf, output], dim=0)
        label = torch.cat([label, target], dim=0)
    return conf, label


model.load_state_dict(torch.load('/Users/rusty1s/Desktop/roc.pt'))


str = ''
for clx in range(0, 30):
    str += 'Class {}:\n'.format(clx)
    conf, label = test(clx)
    _, arg = conf.sort(descending=True)
    sorted_label = label[arg]
    true_count = (label == clx).sum()
    false_count = label.size(0) - true_count

    for i in range(1, label.size(0)-10):
        c1 = (sorted_label[:i] == clx).sum()
        c2 = (sorted_label[:i] != clx).sum()
        str += '{:0.2f}, {:0.6f}\n'.format(c1 / true_count, c2 / false_count)
        if c1 / true_count == 1:
            break

print(str)
with open('/Users/rusty1s/Desktop/roc.txt', 'w') as f:
    f.write(str)
