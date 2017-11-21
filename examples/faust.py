from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import FAUST  # noqa
from torch_geometric.transforms import CartesianAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineConv, Lin  # noqa

path = '~/FAUST'
transform = CartesianAdj()
train_dataset = FAUST(path, train=True, transform=transform)
test_dataset = FAUST(path, train=False, transform=transform)

n = 6890
batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5)
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
input = torch.FloatTensor(n * batch_size, 1).fill_(1)
target = torch.arange(0, n, out=torch.LongTensor()).repeat(batch_size)

if torch.cuda.is_available():
    model, input, target = model.cuda(), input.cuda(), target.cuda()

input, target = Variable(input), Variable(target)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 61:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for data in train_loader:
        adj = data['adj']['content']

        if torch.cuda.is_available():
            adj = adj.cuda()

        optimizer.zero_grad()
        output = model(adj, input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(epoch):
    model.eval()

    correct = 0

    for data in test_loader:
        adj = data['adj']['content']

        if torch.cuda.is_available():
            adj = adj.cuda()

        output = model(adj, input)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    print('Epoch:', epoch, 'Accuracy:', correct / (len(test_dataset) * n))


for epoch in range(1, 101):
    train(epoch)
    test(epoch)
