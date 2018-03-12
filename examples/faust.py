from __future__ import division, print_function

import os.path as osp
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import FAUST  # noqa
from torch_geometric.transform import CartesianAdj  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FAUST')
train_dataset = FAUST(path, train=True, transform=CartesianAdj())
test_dataset = FAUST(path, train=False, transform=CartesianAdj())
train_loader = DataLoader2(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 6890)

    def forward(self, data):
        x, adj = data.input, data.adj
        x = F.elu(self.conv1(adj, x))
        x = F.elu(self.conv2(adj, x))
        x = F.elu(self.conv3(adj, x))
        x = F.elu(self.conv4(adj, x))
        x = F.elu(self.conv5(adj, x))
        x = F.elu(self.conv6(adj, x))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
target = torch.arange(0, model.fc2.weight.size(0)).long()
if torch.cuda.is_available():
    model, target = model.cuda(), target.cuda()

target = Variable(target)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 61:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for data in train_loader:
        data = data.cuda().to_variable('input')
        optimizer.zero_grad()
        F.nll_loss(model(data), target).backward()
        optimizer.step()


def test(epoch):
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.cuda().to_variable('input')
        pred = model(data).data.max(1)[1]
        correct += pred.eq(target.data).sum() / target.size(0)

    print('Epoch:', epoch, 'Test Accuracy:', correct / len(test_dataset))


for epoch in range(1, 101):
    train(epoch)
    test(epoch)
