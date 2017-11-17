from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.graph.grid import grid, grid_position  # noqa
from torch_geometric.transform.graclus import graclus, perm_input  # noqa
from torch_geometric.transform import CartesianAdj  # noqa
from torch_geometric.sparse.stack import stack  # noqa
from torch_geometric.nn.modules import SplineGCN, GraclusMaxPool  # noqa

batch_size = 100
transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '/tmp/MNIST', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '/tmp/MNIST', train=False, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)

adj = grid(torch.Size([28, 28]))
position = grid_position(torch.Size([28, 28]))
adjs, positions, perm = graclus(adj, position, level=4)

adjs = adjs[0:5:2]
positions = positions[0:5:2]
num_first_fc = adjs[2].size(0)

transform = CartesianAdj()
adjs, positions = transform((None, adjs, positions))[1:]

adjs = [stack([a for _ in range(batch_size)])[0] for a in adjs]

if torch.cuda.is_available():
    adjs = [a.cuda() for a in adjs]

adj_0, adj_1, adj_2 = adjs[0], adjs[1], adjs[2]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = GraclusMaxPool(2)
        self.conv1 = SplineGCN(
            1, 32, dim=2, kernel_size=[4, 4], is_open_spline=[1, 1])
        self.conv2 = SplineGCN(
            32, 32, dim=2, kernel_size=[4, 4], is_open_spline=[1, 1])
        self.conv3 = SplineGCN(
            32, 64, dim=2, kernel_size=[4, 4], is_open_spline=[1, 1])
        self.conv4 = SplineGCN(
            64, 64, dim=2, kernel_size=[4, 4], is_open_spline=[1, 1])
        self.fc1 = nn.Linear(num_first_fc * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, adjs, x):
        x = F.elu(self.conv1(adjs[0], x))
        x = F.elu(self.conv2(adjs[0], x))
        x = self.pool(x)
        x = F.elu(self.conv3(adjs[1], x))
        x = F.elu(self.conv4(adjs[1], x))
        x = self.pool(x)
        x = x.contiguous().view(-1, num_first_fc * 64)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
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

    for batch, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1, 1)
        input = torch.cat([perm_input(img, perm) for img in data], dim=0)

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model(adjs, input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(epoch):
    model.eval()

    correct = 0

    for batch, (data, target) in enumerate(test_loader):
        data = data.view(data.size(0), -1, 1)
        input = torch.cat([perm_input(img, perm) for img in data], dim=0)

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()

        input, target = Variable(input), Variable(target)

        output = model(adjs, input)
        pred = output.data.max(1)[1]

        correct += pred.eq(target.data).cpu().sum()

    print('Epoch:', epoch, 'Accuracy:', correct / 10000)


for epoch in range(1, 21):
    train(epoch)
    test(epoch)
