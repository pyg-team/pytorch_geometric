from __future__ import division, print_function

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.graph.grid import grid_5x5, grid_position  # noqa
from torch_geometric.transforms import CartesianAdj  # noqa
from torch_geometric.sparse import stack  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa

batch_size = 100
transform = transforms.Compose([transforms.ToTensor()])
root = os.path.expanduser('~/MNIST')
train_dataset = MNIST(root, train=True, download=True, transform=transform)
test_dataset = MNIST(root, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

size = torch.Size([28, 28])
adj_0 = CartesianAdj()._call(grid_5x5(size), grid_position(size))
size = torch.Size([14, 14])
adj_1 = CartesianAdj()._call(grid_5x5(size), grid_position(size))
adj_0 = stack([adj_0 for _ in range(batch_size)])[0]
adj_1 = stack([adj_1 for _ in range(batch_size)])[0]

if torch.cuda.is_available():
    adj_0, adj_1 = adj_0.cuda(), adj_1.cuda()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, adjs, x):
        # Inference and training takes a long time ...
        # This is due to the `contiguous()` call and the PyTorch `max_pool2d`
        # operation expecting a [N, C, H, W] input format :(
        # However, the convolution calls on GPU are really, really fast.
        x = F.elu(self.conv1(adjs[0], x))
        x = x.view(-1, 28, 28, 32)
        x = x.permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, 32)
        x = F.elu(self.conv2(adjs[1], x))
        x = x.view((-1, 14, 14, 64))
        x = x.permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, 7 * 7 * 64)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.elu(self.fc2(x))
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    if epoch == 11:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for batch, (input, target) in enumerate(train_loader):
        input = input.view(-1, 1)

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model((adj_0, adj_1), input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(epoch):
    model.eval()

    correct = 0

    for batch, (input, target) in enumerate(test_loader):
        input = input.view(-1, 1)

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()

        input = Variable(input)

        output = model((adj_0, adj_1), input)
        pred = output.data.max(1)[1]
        correct += pred.eq(target).cpu().sum()

    print('Epoch:', epoch, 'Accuracy:', correct / len(test_loader))


for epoch in range(1, 21):
    train(epoch)
    test(epoch)
