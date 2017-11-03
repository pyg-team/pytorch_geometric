from __future__ import division, print_function

import time
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixel75  # noqa
from torch_geometric.transform import Graclus, PolarAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineGCN, GraclusMaxPool  # noqa

path = '~/MNISTSuperpixel75'
train_dataset = MNISTSuperpixel75(
    path, train=True, transform=Compose([Graclus(4), PolarAdj()]))
test_dataset = MNISTSuperpixel75(
    path, train=False, transform=Compose([Graclus(4), PolarAdj()]))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineGCN(
            1, 32, dim=2, kernel_size=[3, 8], is_open_spline=[1, 0])
        self.conv2 = SplineGCN(
            32, 64, dim=2, kernel_size=[3, 8], is_open_spline=[1, 0])
        self.conv3 = SplineGCN(
            64, 512, dim=2, kernel_size=[3, 8], is_open_spline=[1, 0])
        self.pool = GraclusMaxPool(2)

    def forward(self, adjs, x):
        x = F.relu(self.conv1(adjs[0], x))
        x = self.pool(x)
        x = F.relu(self.conv2(adjs[1], x))
        x = self.pool(x)
        x = F.relu(self.conv3(adjs[2], x))
        return x


model = Net()
if torch.cuda.is_available():
    model.cuda()


def train(epoch):
    model.train()

    for batch, ((input, adjs, _), target) in enumerate(test_loader):
        adj_0, adj_1, adj_2 = adjs[0][0], adjs[2][0], adjs[4][0]
        input = input.view(-1, 1)

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
            adj_0, adj_1, adj_2 = adj_0.cuda(), adj_1.cuda(), adj_2.cuda()

        t = time.process_time()
        output = model((adj_0, adj_1, adj_2), Variable(input))
        t = time.process_time() - t
        print(t)
        print(output.size())

        #         print(input.size())
        #         print(target.size())
        #         print(adj_0.size(), adj_1.size(), adj_2.size())
        print('batch', batch)


train(1)
# for epoch in range(1, 200):
#     train(epoch)
#     test()
