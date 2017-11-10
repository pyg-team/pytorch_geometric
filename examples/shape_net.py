from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ShapeNet  # noqa
from torch_geometric.transform import EuclideanAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineGCN, Lin  # noqa

path = '~/ShapeNet'
train_dataset = ShapeNet(
    path, train=True, category='cap', transform=EuclideanAdj())
test_dataset = ShapeNet(
    path, train=False, category='cap', transform=EuclideanAdj())

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
num_classes = 2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = SplineGCN(1, 32, dim=3, kernel_size=5)
        self.conv1_2 = SplineGCN(32, 64, dim=3, kernel_size=5)
        self.lin1_3 = Lin(64, 16)
        self.conv2_1 = SplineGCN(16, 32, dim=3, kernel_size=5)
        self.conv2_2 = SplineGCN(32, 64, dim=3, kernel_size=5)
        self.lin2_3 = Lin(64, 16)
        self.conv3_1 = SplineGCN(16, 32, dim=3, kernel_size=5)
        self.conv3_2 = SplineGCN(32, 64, dim=3, kernel_size=5)
        self.lin3_3 = Lin(64, 16)
        self.conv4_1 = SplineGCN(16, 32, dim=3, kernel_size=5)
        self.conv4_2 = SplineGCN(32, 64, dim=3, kernel_size=5)
        self.lin4_3 = Lin(64, 512)
        self.lin4_4 = Lin(512, num_classes)

    def forward(self, adj, x):
        x = F.elu(self.conv1_1(adj, x))
        x = F.elu(self.conv1_2(adj, x))
        x = F.elu(self.lin1_3(x))
        x = F.elu(self.conv2_1(adj, x))
        x = F.elu(self.conv2_2(adj, x))
        x = F.elu(self.lin2_3(x))
        x = F.elu(self.conv3_1(adj, x))
        x = F.elu(self.conv3_2(adj, x))
        x = F.elu(self.lin3_3(x))
        x = F.elu(self.conv4_1(adj, x))
        x = F.elu(self.conv4_2(adj, x))
        x = F.elu(self.lin4_3(x))
        x = F.dropout(x, training=self.training)
        x = self.lin4_4(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 15:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 150:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for batch, ((input, (adj, _), _), target) in enumerate(train_loader):
        if torch.cuda.is_available():
            input, adj, target = input.cuda(), adj.cuda(), target.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model(adj, input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(epoch):
    model.eval()

    acc = 0

    for (input, (adj, _), _), target in test_loader:
        if torch.cuda.is_available():
            input, adj, target = input.cuda(), adj.cuda(), target.cuda()

        output = model(adj, Variable(input))
        pred = output.data.max(1)[1]

        a = 0
        for i in range(num_classes):
            p = pred == i
            t = target == i

            tp = ((p + t) > 1).sum()
            fp = p.sum() - tp
            fn = t.sum() - tp
            a += tp / (tp + fp + fn)
        a /= num_classes
        acc += a

    print('Epoch:', epoch, 'Accuracy:', acc / len(test_loader))


for epoch in range(1, 400):
    train(epoch)
    test(epoch)
