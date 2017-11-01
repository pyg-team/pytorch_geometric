from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import PubMed  # noqa
from torch_geometric.transform import DegreeAdj  # noqa
from torch_geometric.nn.modules import SplineGCN  # noqa

dataset = PubMed('~/PubMed', transform=DegreeAdj())
data, target = dataset[0]
input, adj, _ = data
train_mask, test_mask = dataset.train_mask, dataset.test_mask

if torch.cuda.is_available():
    input, adj = dataset.input.cuda(), dataset.adj.cuda()
    target = dataset.target.cuda()
    train_mask, test_mask = train_mask.cuda(), test_mask.cuad()

input, target = Variable(input), Variable(target)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineGCN(500, 16, dim=2, kernel_size=[2, 2])
        self.conv2 = SplineGCN(16, 7, dim=2, kernel_size=[2, 2])

    def forward(self, adj, x):
        x = F.relu(self.conv1(adj, x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(adj, x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)


def train(epoch):
    model.train()

    optimizer.zero_grad()
    output = model(adj, input)
    output = output[train_mask]
    loss = F.nll_loss(output, target, size_average=True)
    loss.backward()
    optimizer.step()

    print('Epoch:', epoch, 'Loss:', loss.data[0])


def test():
    model.eval()

    output = model(adj, input)
    output = output[test_mask]
    pred = output.data.max(1)[1]
    acc = pred.eq(target.data).sum()

    print('Accuracy:', acc / output.size(0))


for epoch in range(1, 200):
    train(epoch)
    test()
