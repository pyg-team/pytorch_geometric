from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import FAUST  # noqa
from torch_geometric.transform import PolarAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineGCN, Lin  # noqa
from torch_geometric.validation import max_geodesic_error_accuracy  # noqa

path = '~/MPI-FAUST'
train_dataset = FAUST(
    path, train=True, correspondence=True, transform=PolarAdj())
test_dataset = FAUST(
    path, train=False, correspondence=True, transform=PolarAdj())

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


# 85.39% after 99 epochs, euclidean adj, [5, 5, 2], [1, 1, 1], 4x conv
# 87.53% after 99 epochs, polar adj, [3, 4, 3], [1, 0, 1], 4x conv
# 89.26% after 99 epochs, polar adj, [3, 4, 3], [1, 0, 1], 3x conv + lr decay
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineGCN(
            1, 32, dim=3, kernel_size=[3, 4, 3], is_open_spline=[1, 0, 1])
        self.conv2 = SplineGCN(
            32, 64, dim=3, kernel_size=[3, 4, 3], is_open_spline=[1, 0, 1])
        self.conv3 = SplineGCN(
            64, 128, dim=3, kernel_size=[3, 4, 3], is_open_spline=[1, 0, 1])
        self.lin1 = Lin(128, 256)
        self.lin2 = Lin(256, 6890)

    def forward(self, adj, x):
        x = F.relu(self.conv1(adj, x))
        x = F.relu(self.conv2(adj, x))
        x = F.relu(self.conv3(adj, x))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    # Learning rate decay after 60 epochs.
    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    # Learning rate decay after 100 epochs.
    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    # Learning rate decay after 200 epochs.
    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    for batch, ((_, (adj, _), _), target) in enumerate(train_loader):
        input, target = torch.ones(adj.size(0), 1), target.view(-1)

        if torch.cuda.is_available():
            adj, input, target = adj.cuda(), input.cuda(), target.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model(adj, input)
        loss = F.nll_loss(output, target, size_average=True)
        loss.backward()
        optimizer.step()

        print('Epoch:', epoch, 'Batch:', batch, 'Loss:', loss.data[0])


def test():
    model.eval()

    acc_0 = acc_1 = acc_2 = acc_4 = acc_8 = 0

    for (_, (adj, _), position), target in test_loader:
        input = torch.ones(adj.size(0), 1)
        p, target = position.view(-1, 3), target.view(-1)

        if torch.cuda.is_available():
            input, adj, target = input.cuda(), adj.cuda(), target.cuda()
            p = p.cuda()

        output = model(adj, Variable(input))

        pred = output.data.max(1)[1]

        acc_0 += max_geodesic_error_accuracy(p, pred, target, error=0.0)
        acc_1 += max_geodesic_error_accuracy(p, pred, target, error=0.01)
        acc_2 += max_geodesic_error_accuracy(p, pred, target, error=0.02)
        acc_4 += max_geodesic_error_accuracy(p, pred, target, error=0.04)
        acc_8 += max_geodesic_error_accuracy(p, pred, target, error=0.08)

    print('Accuracy 0:', acc_0 / (20 * 6890))
    print('Accuracy 1:', acc_1 / (20 * 6890))
    print('Accuracy 2:', acc_2 / (20 * 6890))
    print('Accuracy 4:', acc_4 / (20 * 6890))
    print('Accuracy 8:', acc_8 / (20 * 6890))


for epoch in range(1, 300):
    train(epoch)
    test()
