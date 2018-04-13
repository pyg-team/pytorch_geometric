from __future__ import division, print_function

import os.path as osp
import sys

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_scatter import scatter_mean

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ENZYMES  # noqa
from torch_geometric.transform import TargetIndegreeAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ENZYMES')
train_dataset = ENZYMES(path, transform=TargetIndegreeAdj())
test_dataset = ENZYMES(path, transform=TargetIndegreeAdj())
perm = torch.randperm(len(train_dataset))
train_dataset.split = perm[len(train_dataset) // 10:]
test_dataset.split = perm[:len(test_dataset) // 10]
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(3, 64, dim=1, kernel_size=2)
        self.conv2 = SplineConv(64, 64, dim=1, kernel_size=2)
        self.conv3 = SplineConv(64, 64, dim=1, kernel_size=2)
        self.conv4 = SplineConv(64, 64, dim=1, kernel_size=2)
        self.fc1 = nn.Linear(64, 6)

    def forward(self, data):
        x = F.elu(self.conv1(data.adj, data.input))
        x = F.elu(self.conv2(data.adj, x))
        x = F.elu(self.conv3(data.adj, x))
        x = F.elu(self.conv4(data.adj, x))

        index = Variable(data.batch.unsqueeze(1).expand(x.size()))
        x = scatter_mean(index, x, dim=0)

        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.target)
        loss.backward()
        optimizer.step()


def test(epoch, dataset, loader, str):
    model.eval()
    correct = 0

    for data in loader:
        data = data.cuda().to_variable('input')
        pred = model(data).data.max(1)[1]
        correct += pred.eq(data.target).sum()

    print('Epoch:', epoch, str, 'Accuracy:', correct / len(dataset))


for epoch in range(1, 501):
    train()
    # test(epoch, train_dataset, train_loader, 'Train')
    test(epoch, test_dataset, test_loader, 'Test')
