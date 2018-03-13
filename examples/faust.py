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
from torch_geometric.datasets.utils.download import download_url  # noqa
from torch_geometric.datasets.utils.extract import extract_tar  # noqa

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FAUST')
train_dataset = FAUST(path, train=True, transform=CartesianAdj())
test_dataset = FAUST(path, train=False, transform=CartesianAdj())
train_loader = DataLoader2(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=1)

#url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/' \
#      'faust_geodesic_distance.tar.gz'
#download_url(url, path)
#extract_tar(osp.join(path, 'faust_geodesic_distance.tar.gz'), path)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.conv2 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv7 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv8 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv9 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv10 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv11 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv12 = SplineConv(64, 64, dim=3, kernel_size=5)
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
        x = F.elu(self.conv7(adj, x))
        x = F.elu(self.conv8(adj, x))
        x = F.elu(self.conv9(adj, x))
        x = F.elu(self.conv10(adj, x))
        x = F.elu(self.conv11(adj, x))
        x = F.elu(self.conv12(adj, x))
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
    acc_0 = acc_1 = acc_2 = acc_4 = acc_6 = acc_8 = acc_10 = 0

    for i, data in enumerate(test_loader):
        data = data.cuda().to_variable('input')
        pred = model(data).data.max(1)[1]
        filename = '{}.pt'.format(80 + i)
        distance = torch.load(osp.join(path, 'geodesic_distance', filename))
        distance = distance.cuda()
        geodesic_error = distance[pred, target.data]
        acc_0 += (geodesic_error <= 0.0000002).sum()
        acc_1 += (geodesic_error <= 0.01).sum()
        acc_2 += (geodesic_error <= 0.02).sum()
        acc_4 += (geodesic_error <= 0.04).sum()
        acc_6 += (geodesic_error <= 0.06).sum()
        acc_8 += (geodesic_error <= 0.08).sum()
        acc_10 += (geodesic_error <= 0.1).sum()

    print('Epoch', epoch)
    print('Accuracy 0:', acc_0 / (20 * 6890))
    print('Accuracy 1:', acc_1 / (20 * 6890))
    print('Accuracy 2:', acc_2 / (20 * 6890))
    print('Accuracy 4:', acc_4 / (20 * 6890))
    print('Accuracy 6:', acc_6 / (20 * 6890))
    print('Accuracy 8:', acc_8 / (20 * 6890))
    print('Accuracy 10:', acc_10 / (20 * 6890))


for epoch in range(1, 101):
    train(epoch)
    test(epoch)
