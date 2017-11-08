from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixel75  # noqa
from torch_geometric.transform import Graclus, PolarAdj, EuclideanAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineGCN, GraclusMaxPool  # noqa
from torch_geometric.nn.functional import batch_average  # noqa

path = '~/MNISTSuperpixel75'
train_dataset = MNISTSuperpixel75(
    path, train=True, transform=Compose([Graclus(4),
                                         EuclideanAdj()]))
test_dataset = MNISTSuperpixel75(
    path, train=False, transform=Compose([Graclus(4),
                                          EuclideanAdj()]))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = GraclusMaxPool(2)
        self.conv1 = SplineGCN(
            2, 32, dim=2, kernel_size=[5, 5], is_open_spline=[1, 1])
        self.conv2 = SplineGCN(
            32, 64, dim=2, kernel_size=[5, 5], is_open_spline=[1, 1])
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

conv_1_weight = model.conv1.weight.data.cpu()
conv_2_weight = model.conv2.weight.data.cpu()
torch.save(conv_1_weight, '/tmp/conv_1_weight.pt')
torch.save(conv_2_weight, '/tmp/conv_2_weight.pt')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 11:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for batch, ((input, adjs, _), target) in enumerate(train_loader):
        adj_0, adj_1, adj_2 = adjs[0][0], adjs[2][0], adjs[4][0]
        ones = input.new(input.size(0)).fill_(1)
        input = torch.stack([input, ones], dim=1)
        slice = adjs[4][1][:, 0]

        if torch.cuda.is_available():
            input, target, slice = input.cuda(), target.cuda(), slice.cuda()
            adj_0, adj_1, adj_2 = adj_0.cuda(), adj_1.cuda(), adj_2.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model((adj_0, adj_1, adj_2), input, slice)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(epoch):
    model.eval()

    correct = 0

    for batch, ((input, adjs, _), target) in enumerate(test_loader):
        adj_0, adj_1, adj_2 = adjs[0][0], adjs[2][0], adjs[4][0]
        ones = input.new(input.size(0)).fill_(1)
        input = torch.stack([input, ones], dim=1)
        slice = adjs[4][1][:, 0]

        if torch.cuda.is_available():
            input, target, slice = input.cuda(), target.cuda(), slice.cuda()
            adj_0, adj_1, adj_2 = adj_0.cuda(), adj_1.cuda(), adj_2.cuda()

        input, target = Variable(input), Variable(target)

        output = model((adj_0, adj_1, adj_2), input, slice)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    print('Epoch:', epoch, 'Accuracy:', correct / 10000)


for epoch in range(1, 21):
    train(epoch)
    test(epoch)
