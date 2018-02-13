import os
import sys

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ShapeNet  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import CartesianAdj  # noqa
from torch_geometric.nn.modules import SplineConv, Lin  # noqa
from torch_geometric.nn.functional import voxel_max_pool  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ShapeNet')

category = 'Cap'
num_classes = 3
transform = CartesianAdj()
train_dataset = ShapeNet(path, category, True, transform)
test_dataset = ShapeNet(path, category, False, transform)
train_loader = DataLoader2(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=16)


def upscale(input, cluster):
    cluster = Variable(cluster)
    cluster = cluster.view(-1, 1).expand(cluster.size(0), input.size(1))
    return torch.gather(input, 0, cluster)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.fc1 = Lin(64, 64)
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv7 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.fc2 = Lin(64, num_classes)

    def forward(self, data1):
        # print('Num nodes', data1.num_nodes)
        data1.input = F.elu(self.conv1(data1.adj, data1.input))
        data2, cluster1 = voxel_max_pool(data1, 0.03, transform)
        # print('First pool', data2.num_nodes)
        data2.input = F.elu(self.conv2(data2.adj, data2.input))
        data3, cluster2 = voxel_max_pool(data2, 0.05, transform)
        # print('Second pool', data3.num_nodes)
        data3.input = F.elu(self.conv3(data3.adj, data3.input))
        data4, cluster3 = voxel_max_pool(data3, 0.07, transform)
        # print('Third pool', data4.num_nodes)
        data4.input = F.elu(self.conv4(data4.adj, data4.input))
        data4.input = F.elu(self.fc1(data4.input))
        data3.input = upscale(data4.input, cluster3)
        data3.input = F.elu(self.conv5(data3.adj, data3.input))
        data2.input = upscale(data3.input, cluster2)
        data2.input = F.elu(self.conv6(data2.adj, data2.input))
        data1.input = upscale(data2.input, cluster1)
        data1.input = F.elu(self.conv7(data1.adj, data1.input))
        data1.input = F.elu(self.fc2(data1.input))
        return F.log_softmax(data1.input, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.target)
        loss.backward()
        optimizer.step()


def test(epoch):
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.cuda().to_variable()
        output = model(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(data.target.data).cpu().sum()

    num_nodes = test_dataset.set.input.size(0)
    print('Epoch:', epoch, 'Test Accuracy:', correct / num_nodes)


for epoch in range(1, 31):
    train(epoch)
    test(epoch)
