from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cora  # noqa
from torch_geometric.transform import DegreeAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineGCN  # noqa

path = '~/Cora'
train_dataset = Cora(path, train=True, transform=DegreeAdj())
test_dataset = Cora(path, train=False, transform=DegreeAdj())

train_loader = DataLoader(train_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineGCN(1433, 32, dim=2, kernel_size=[3, 3])
        self.conv2 = SplineGCN(32, 7, dim=2, kernel_size=[3, 3])

    def forward(self, adj, x):
        x = F.relu(self.conv1(adj, x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(adj, x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    for batch, ((input, (adj, _)), target, mask) in enumerate(train_loader):
        input, target, mask = input.squeeze(), target.squeeze(), mask.squeeze()

        if torch.cuda.is_available():
            input, adj = input.cuda(), adj.cuda()
            target, mask = target.cuda(), mask.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model(adj, input)
        output = output[mask]
        loss = F.nll_loss(output, target, size_average=True)
        loss.backward()
        optimizer.step()

        print('Epoch:', epoch, 'Batch:', batch, 'Loss:', loss.data[0])


def test():
    model.eval()

    for (input, (adj, _)), target, mask in test_loader:
        input, target, mask = input.squeeze(), target.squeeze(), mask.squeeze()

        if torch.cuda.is_available():
            input, adj = input.cuda(), adj.cuda()
            target, mask = target.cuda(), mask.cuda()

        input, target = Variable(input), Variable(target)

        output = model(adj, input)
        output = output[mask]

        pred = output.data.max(1)[1]
        acc = pred.eq(target.data).sum()

        print('Accuracy:', acc / output.size(0))


for epoch in range(1, 200):
    train(epoch)
    test()
