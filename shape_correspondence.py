from __future__ import division, print_function

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch_geometric.datasets.faust import FAUST
from torch_geometric.graph.geometry import EuclideanAdj
from torch_geometric.utils.dataloader import DataLoader
from torch_geometric.nn.modules import SplineGCN, Lin

path = '~/MPI-FAUST'
train_dataset = FAUST(
    path, train=True, correspondence=True, transform=EuclideanAdj())
test_dataset = FAUST(
    path, train=False, correspondence=True, transform=EuclideanAdj())

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


# Reaches 85.39% accuracy after 99 epochs.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineGCN(
            1, 32, dim=3, kernel_size=[5, 5, 2], is_open_spline=True)
        self.conv2 = SplineGCN(
            32, 64, dim=3, kernel_size=[5, 5, 2], is_open_spline=True)
        self.conv3 = SplineGCN(
            64, 64, dim=3, kernel_size=[5, 5, 2], is_open_spline=True)
        self.conv4 = SplineGCN(
            64, 128, dim=3, kernel_size=[5, 5, 2], is_open_spline=True)
        self.lin1 = Lin(128, 256)
        self.lin2 = Lin(256, 6890)

    def forward(self, adj, x):
        x = F.relu(self.conv1(adj, x))
        x = F.relu(self.conv2(adj, x))
        x = F.relu(self.conv3(adj, x))
        x = F.relu(self.conv4(adj, x))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    for batch_idx, ((_, (adj, _)), target) in enumerate(train_loader):
        input = torch.ones(adj.size(0)).view(-1, 1)

        if torch.cuda.is_available():
            input, adj, target = input.cuda(), adj.cuda(), target.cuda()

        input, target = Variable(input), Variable(target)
        output = model(adj, input)

        optimizer.zero_grad()
        loss = F.nll_loss(output, target.view(-1), size_average=True)
        loss.backward()
        optimizer.step()
        loss = loss.data.cpu()[0]

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        print('Epoch: ', epoch, 'Batch: ', batch_idx, 'Loss: ', loss,
              'Correct:', correct)


def test():
    model.eval()
    correct = 0

    for (_, (adj, _)), target in test_loader:
        input = torch.ones(adj.size(0)).view(-1, 1)

        if torch.cuda.is_available():
            input, adj, target = input.cuda(), adj.cuda(), target.cuda()

        output = model(adj, Variable(input))

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum()

    print('Accuracy:', correct / (20 * 6890))


for epoch in range(1, 100):
    train(epoch)
    test()
