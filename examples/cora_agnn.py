from __future__ import division, print_function

import os.path as osp
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cora  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.nn.modules import AGNN  # noqa

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = Cora(osp.join(path, 'Cora'), normalize=True)
data = dataset[0].cuda().to_variable()
num_features, num_targets = data.input.size(1), data.target.data.max() + 1
train_mask = torch.arange(0, 20 * num_targets).long()
val_mask = torch.arange(train_mask.size(0), train_mask.size(0) + 500).long()
test_mask = torch.arange(data.num_nodes - 1000, data.num_nodes).long()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.conv1 = AGNN(requires_grad=False)
        self.conv2 = AGNN(requires_grad=True)
        self.fc2 = nn.Linear(16, num_targets)

    def forward(self):
        x = F.dropout(data.input, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.conv1(x, data.index)
        x = self.conv2(x, data.index)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    train_mask, val_mask = train_mask.cuda(), val_mask.cuda()
    test_mask, model = test_mask.cuda(), model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[train_mask], data.target[train_mask]).backward()
    optimizer.step()


def test(mask):
    model.eval()
    pred = model()[mask].data.max(1)[1]
    return pred.eq(data.target.data[mask]).sum() / mask.size(0)


acc = []
for run in range(1, 101):
    model.fc1.reset_parameters()
    model.conv1.reset_parameters()
    model.conv2.reset_parameters()
    model.fc2.reset_parameters()

    old_val = 0
    cur_test = 0
    for _ in range(1000):
        train()
        val = test(val_mask)
        if val > old_val:
            old_val = val
            cur_test = test(test_mask)

    acc.append(cur_test)
    print('Run:', run, 'Test Accuracy:', acc[-1])

print('Mean:', torch.Tensor(acc).mean(), 'Stddev:', torch.Tensor(acc).std())
