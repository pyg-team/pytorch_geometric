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
from torch_geometric.nn.modules import SplineGCN  # noqa

dataset = Cora('~/Cora', transform=DegreeAdj())
(input, adj, _), target = dataset[0]
train_mask, test_mask = dataset.train_mask, dataset.test_mask

# one = input.new(input.size(0)).fill_(1)
# input = torch.cat([input, one], dim=1)

if torch.cuda.is_available():
    input, adj, target = input.cuda(), adj.cuda(), target.cuda()
    train_mask, test_mask = train_mask.cuda(), test_mask.cuda()

input, target = Variable(input), Variable(target)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineGCN(1433, 16, dim=2, kernel_size=[2, 2], bias=False)
        self.conv2 = SplineGCN(16, 7, dim=2, kernel_size=[2, 2], bias=False)

    def forward(self, adj, x):
        x = F.elu(self.conv1(adj, x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(adj, x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)


def train(epoch):
    model.train()

    optimizer.zero_grad()
    output = model(adj, input)[train_mask]
    loss = F.nll_loss(output, target[train_mask], size_average=True)
    loss.backward()
    optimizer.step()


def test(epoch):
    model.eval()

    output = model(adj, input)
    output = output[test_mask]
    pred = output.data.max(1)[1]
    acc = pred.eq(target.data[test_mask]).sum() / output.size(0)

    return acc


accs = []
for run in range(0, 201):
    model.conv1.reset_parameters()
    model.conv2.reset_parameters()
    for epoch in range(1, 200):
        train(epoch)
        acc = test(epoch)
    print('Run:', run, 'Accuracy:', acc)
    accs.append(acc)

accs = torch.FloatTensor(accs)
print('Mean:', accs.mean(), 'Stddev:', accs.std())
