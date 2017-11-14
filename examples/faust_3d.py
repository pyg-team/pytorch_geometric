from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import FAUST  # noqa
from torch_geometric.transform import PolarAdj, EuclideanAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineGCN, Lin  # noqa

path = '~/MPI-FAUST'
train_dataset = FAUST(
    path, train=True, correspondence=True, transform=EuclideanAdj())
test_dataset = FAUST(
    path, train=False, correspondence=True, transform=EuclideanAdj())

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineGCN(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineGCN(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv5 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.lin1 = Lin(64, 256)
        self.lin2 = Lin(256, 6890)

    def forward(self, adj, x):
        x = F.elu(self.conv1(adj, x))
        x = F.elu(self.conv2(adj, x))
        x = F.elu(self.conv3(adj, x))
        x = F.elu(self.conv4(adj, x))
        x = F.elu(self.conv5(adj, x))
        x = F.elu(self.conv6(adj, x))
        x = F.elu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

input = torch.FloatTensor(6890, 1).fill_(1)
if torch.cuda.is_available():
    input = input.cuda()
input = Variable(input)


def train(epoch):
    model.train()

    if epoch == 61:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 121:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for batch, ((_, (adj, _), _), target) in enumerate(train_loader):
        if torch.cuda.is_available():
            adj, target, = adj.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(adj, input)
        loss = F.nll_loss(output, Variable(target))
        loss.backward()
        optimizer.step()

        print('Epoch:', epoch, 'Batch:', batch, 'Loss:', loss.data[0])


def test():
    model.eval()

    acc_0 = acc_1 = acc_2 = acc_4 = acc_6 = acc_8 = acc_10 = 0

    for (_, (adj, _), _), target, distance in test_loader:
        if torch.cuda.is_available():
            adj, target, distance = adj.cuda(), target.cuda(), distance.cuda()

        target = Variable(target)

        output = model(adj, input)
        pred = output.data.max(1)[1]
        geodesic_error = distance[pred, target.data]
        acc_0 += (geodesic_error <= 0.0000002).sum()
        acc_1 += (geodesic_error <= 0.01).sum()
        acc_2 += (geodesic_error <= 0.02).sum()
        acc_4 += (geodesic_error <= 0.04).sum()
        acc_6 += (geodesic_error <= 0.06).sum()
        acc_8 += (geodesic_error <= 0.08).sum()
        acc_10 += (geodesic_error <= 0.1).sum()

    print('Accuracy 0:', acc_0 / (20 * 6890))
    print('Accuracy 1:', acc_1 / (20 * 6890))
    print('Accuracy 2:', acc_2 / (20 * 6890))
    print('Accuracy 4:', acc_4 / (20 * 6890))
    print('Accuracy 6:', acc_6 / (20 * 6890))
    print('Accuracy 8:', acc_8 / (20 * 6890))
    print('Accuracy 10:', acc_10 / (20 * 6890))


for epoch in range(1, 151):
    train(epoch)
    test()
