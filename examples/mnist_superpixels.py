from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixels  # noqa
from torch_geometric.transforms import Graclus, CartesianAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineConv, GraclusMaxPool  # noqa
from torch_geometric.nn.functional import batch_average  # noqa

path = '~/MNISTSuperpixels'
transform = Compose([Graclus(4), CartesianAdj()])
train_dataset = MNISTSuperpixels(path, train=True, transform=transform)
test_dataset = MNISTSuperpixels(path, train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = GraclusMaxPool(2)
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, adjs, x, slice):
        x = F.elu(self.conv1(adjs[0], x))
        x = self.pool(x)
        x = F.elu(self.conv2(adjs[1], x))
        x = self.pool(x)
        x = batch_average(x, slice)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 11:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for batch, data in enumerate(train_loader):
        input, target = data['input'].view(-1, 1), data['target']
        adj_0, adj_1 = data['adjs'][0]['content'], data['adjs'][2]['content']
        slice = data['adjs'][4]['slice'][:, 0]

        if torch.cuda.is_available():
            input, target, slice = input.cuda(), target.cuda(), slice.cuda()
            adj_0, adj_1 = adj_0.cuda(), adj_1.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model((adj_0, adj_1), input, slice)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(epoch):
    model.eval()

    correct = 0

    for batch, data in enumerate(test_loader):
        input, target = data['input'].view(-1, 1), data['target']
        adj_0, adj_1 = data['adjs'][0]['content'], data['adjs'][2]['content']
        slice = data['adjs'][4]['slice'][:, 0]

        if torch.cuda.is_available():
            input, target, slice = input.cuda(), target.cuda(), slice.cuda()
            adj_0, adj_1 = adj_0.cuda(), adj_1.cuda()

        input = Variable(input)

        output = model((adj_0, adj_1), input, slice)
        pred = output.data.max(1)[1]
        correct += pred.eq(target).cpu().sum()

    print('Epoch:', epoch, 'Accuracy:', correct / len(test_dataset))


for epoch in range(1, 21):
    train(epoch)
    test(epoch)
