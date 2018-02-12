import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ShapeNet  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.transform import CartesianAdj  # noqa

import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean
from torch.autograd import Variable
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.modules import Lin  # noqa
from torch_geometric.nn.functional import max_pool  # noqa
from torch_geometric.sparse import SparseTensor  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ShapeNet')

category = 'Cap'
transform = CartesianAdj()
train_dataset = ShapeNet(path, category, True, transform)
test_dataset = ShapeNet(path, category, False, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

num_classes = 2

def to_cartesian_adj(index, position):
    row, col = index
    n, dim = position.size()
    cartesian = position[col] - position[row]
    cartesian *= 1 / (2 * cartesian.abs().max())
    cartesian += 0.5
    return SparseTensor(index, cartesian, torch.Size([n, n, dim]))


def max_grid_pool(x, position, adj, size):
    grid_size = position.new(2).fill_(size)
    cluster = grid_cluster(position, grid_size)
    x, index, position = max_pool(x, adj._indices(), position, cluster)
    adj = to_cartesian_adj(index, position)
    return x, position, adj, cluster


def upscaling(input_features,index):
    #output = input_features.data.new(index.size(0),input_features.size(1))
    #output = Variable(output)
    return torch.gather(input_features, 0, index)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.fc1 = Lin(64,64)
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv7 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.fc2 = Lin(64,num_classes)


    def forward(self, x, position, adj):
        x = x.unsqueeze(1)
        x = F.elu(self.conv1(adj, x))
        x, position, adj1, cluster1 = max_grid_pool(x, position, adj, 3)
        x = F.elu(self.conv2(adj1, x))
        x, position, adj2, cluster2 = max_grid_pool(x, position, adj1, 5)
        x = F.elu(self.conv3(adj2, x))
        x, position, adj3, cluster3 = max_grid_pool(x, position, adj2, 10)
        x = F.elu(self.conv4(adj3, x))
        x = F.elu(self.fc1(x))
        x = upscaling(x, cluster3)
        x = F.elu(self.conv5(adj3, x))
        x = upscaling(x, cluster2)
        x = F.elu(self.conv6(adj2, x))
        x = upscaling(x, cluster1)
        x = F.elu(self.conv7(adj1, x))
        x = F.elu(self.fc2(adj1, x))



        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        output = model(data.input, data.pos, data.adj)
        loss = F.nll_loss(output, data.target)
        loss.backward()
        optimizer.step()

def test(epoch):
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.cuda().to_variable()

        output = model(data.input, data.pos, data.adj)
        pred = output.data.max(1)[1]
        correct += pred.eq(data.target.data).cpu().sum()

    print('Epoch:', epoch, 'Test Accuracy:', correct / len(test_dataset))


for epoch in range(1, 31):
    train(epoch)
    test(epoch)
