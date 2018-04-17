from __future__ import division, print_function

import os.path as osp

import torch
from torch.autograd import Variable as Var
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import Cora
from torch_geometric.nn.modules import RandomWalk
from torch_geometric.utils import one_hot, degree

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = Cora(osp.join(path, 'Cora'))
data = dataset[0].cuda() if torch.cuda.is_available() else dataset[0]
x, edge_index, target = data.input, data.index, data.target
row, col = edge_index
num_nodes, num_features, num_classes = x.size(0), x.size(1), target.max() + 1
one_hot = one_hot(target, num_classes)
edge_attr = (1 / degree(row, num_nodes))[col]

train_mask = torch.arange(0, 20 * num_classes).long()
val_mask = torch.arange(train_mask.size(0), train_mask.size(0) + 500).long()
test_mask = torch.arange(num_nodes - 1000, num_nodes).long()

one_hot[train_mask[-1]:, :] = 0

edge_attr, one_hot, target = Var(edge_attr), Var(one_hot), Var(target)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.walk = RandomWalk(num_classes, 30)

    def forward(self):
        x = self.walk(edge_index, edge_attr, one_hot)
        return F.log_softmax(x, dim=1)


model = Net().cuda() if torch.cuda.is_available() else Net()

train_mask, val_mask = train_mask.cuda(), val_mask.cuda()
test_mask = test_mask.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[train_mask], target[train_mask]).backward()
    optimizer.step()


def test(mask):
    model.eval()
    pred = model()[mask].data.max(1)[1]
    return pred.eq(target.data[mask]).sum() / mask.size(0)


for epoch in range(1, 1001):
    train()
    train_acc = test(train_mask)
    val_acc = test(val_mask)
    test_acc = test(test_mask)
    print(epoch, train_acc, val_acc, test_acc)
