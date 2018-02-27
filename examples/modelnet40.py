import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet40  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import (RandomRotate, RandomScale,
                                       RandomTranslate, RandomShear,
                                       NormalizeScale)  # noqa
from torch_geometric.transform import CartesianAdj, CartesianLocalAdj, \
    SphericalAdj, Spherical2dAdj  # noqa
from torch_geometric.nn.modules import SplineConv, Lin  # noqa
from torch_geometric.nn.functional import voxel_max_pool  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10')

transform = Compose([
    RandomRotate(3.14),
    #RandomScale(1.1),
    #RandomShear(0.5),
    #RandomTranslate(0.005),
    NormalizeScale(),
    CartesianAdj()
])

transform_test = Compose([
    #    RandomRotate(3.14),
    #RandomScale(1.1),
    #RandomShear(0.5),
    #RandomTranslate(0.005),
    NormalizeScale(),
    CartesianAdj()
])

trans = CartesianAdj()

batch_size = 1
train_dataset = ModelNet40(path, True, transform=transform)
test_dataset = ModelNet40(path, False, transform=transform_test)
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size, shuffle=True)
num_classes = 40

#single_example = train_dataset[3713]


def show_pointclouds(data):
    data.batch = torch.zeros(data.num_nodes).long()
    data.target = None
    data = data.cuda().to_variable()
    #output, data2, data3, data4 = model(data)
    #pred = output.data.max(1)[1]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax_gt = fig.add_subplot(3, 2, 2, projection='3d')
    ax_sub1 = fig.add_subplot(3, 2, 3, projection='3d')
    ax_sub2 = fig.add_subplot(3, 2, 4, projection='3d')
    ax_sub3 = fig.add_subplot(3, 2, 5, projection='3d')
    data2 = data
    data3 = data
    data4 = data
    ax.scatter(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2])
    #ax.axis('equal')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)

    ax_sub1.scatter(data2.pos[:, 0], data2.pos[:, 1], data2.pos[:, 2])
    for i in range(data2.adj._indices().size(1)):
        idx = data2.adj._indices()[:, i]
        ax_sub1.plot(
            [data2.pos[idx[0]][0], data2.pos[idx[1]][0]],
            [data2.pos[idx[0]][1], data2.pos[idx[1]][1]],
            zs=[data2.pos[idx[0]][2], data2.pos[idx[1]][2]])
    #ax_sub1.axis('equal')
    ax_sub1.set_xlim3d(0, 1)
    ax_sub1.set_ylim3d(0, 1)
    ax_sub1.set_zlim3d(0, 1)

    ax_sub2.scatter(data3.pos[:, 0], data3.pos[:, 1], data3.pos[:, 2])
    for i in range(data3.adj._indices().size(1)):
        idx = data3.adj._indices()[:, i]
        ax_sub2.plot(
            [data3.pos[idx[0]][0], data3.pos[idx[1]][0]],
            [data3.pos[idx[0]][1], data3.pos[idx[1]][1]],
            zs=[data3.pos[idx[0]][2], data3.pos[idx[1]][2]])
    #ax_sub2.axis('equal')
    ax_sub2.set_xlim3d(0, 1)
    ax_sub2.set_ylim3d(0, 1)
    ax_sub2.set_zlim3d(0, 1)

    ax_sub3.scatter(data4.pos[:, 0], data4.pos[:, 1], data4.pos[:, 2])
    for i in range(data4.adj._indices().size(1)):
        idx = data4.adj._indices()[:, i]
        ax_sub3.plot(
            [data4.pos[idx[0]][0], data4.pos[idx[1]][0]],
            [data4.pos[idx[0]][1], data4.pos[idx[1]][1]],
            zs=[data4.pos[idx[0]][2], data4.pos[idx[1]][2]])
    #ax_sub3.axis('equal')
    ax_sub3.set_xlim3d(0, 1)
    ax_sub3.set_ylim3d(0, 1)
    ax_sub3.set_zlim3d(0, 1)

    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(33, 64, dim=3, kernel_size=5)
        self.conv22 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv32 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(65, 96, dim=3, kernel_size=5)
        self.conv42 = SplineConv(96, 96, dim=3, kernel_size=5)
        self.conv5 = SplineConv(97, 96, dim=3, kernel_size=5)
        self.conv52 = SplineConv(96, 96, dim=3, kernel_size=5)
        self.conv6 = SplineConv(97, 128, dim=3, kernel_size=5)
        self.conv62 = SplineConv(128, 128, dim=3, kernel_size=5)
        self.conv7 = SplineConv(128, 128, dim=3, kernel_size=5)
        self.conv72 = SplineConv(128, 128, dim=3, kernel_size=5)
        #self.conv8 = SplineConv(256, 256, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(128, num_classes)

        self.skip1 = Lin(32, 64)
        self.skip2 = Lin(64, 96)
        self.skip3 = Lin(96, 128)
        self.skip4 = Lin(128, 256)
        self.skip5 = Lin(256, 256)

    def forward(self, data):
        ones = Variable(data.input.data.new(data.input.size(0), 1).fill_(1))
        #print('Num nodes', data.num_nodes)
        data.input = F.elu(self.conv1(data.adj, data.input))
        #:data.input = self.bn1(data.input)
        data, _ = voxel_max_pool(data, (1 / 32), origin=0, transform=trans)
        x = torch.cat([data.input, ones[:data.input.size(0)]], dim=1)
        x = F.elu(self.conv2(data.adj, x))
        x = self.conv22(data.adj, x)
        data.input = F.elu(x + self.skip1(data.input))
        #data.input = self.bn2(data.input)
        #data, _ = voxel_max_pool(data, (1/32), origin=0, transform=trans)
        #data_p1 = data
        #x = torch.cat((data.input,ones[:data.input.size(0),:]),dim=1)
        x = F.elu(self.conv3(data.adj, data.input))
        x = self.conv32(data.adj, x)
        data.input = F.elu(x + data.input)
        #data.input = self.bn3(data.input)
        data, _ = voxel_max_pool(data, (1 / 16), origin=0, transform=trans)
        x = torch.cat([data.input, ones[:data.input.size(0)]], dim=1)
        x = F.elu(self.conv4(data.adj, x))
        x = self.conv42(data.adj, x)
        data.input = F.elu(x + self.skip2(data.input))
        #data.input = self.bn4(data.input)
        data, _ = voxel_max_pool(data, (1 / 8), origin=0, transform=trans)
        x = torch.cat([data.input, ones[:data.input.size(0)]], dim=1)
        x = F.elu(self.conv5(data.adj, x))
        x = self.conv52(data.adj, x)
        data.input = F.elu(x + data.input)
        #data.input = self.bn5(data.input)
        data, _ = voxel_max_pool(data, (1 / 4), origin=0, transform=trans)
        x = torch.cat([data.input, ones[:data.input.size(0)]], dim=1)
        x = F.elu(self.conv6(data.adj, x))
        x = self.conv62(data.adj, x)
        data.input = F.elu(x + self.skip3(data.input))
        #data, _ = voxel_max_pool(data, (1/2), origin=0, transform=trans)
        #x = torch.cat([data.input, ones[:data.input.size(0)]], dim=1)
        x = F.elu(self.conv7(data.adj, data.input))
        x = self.conv72(data.adj, x)
        data.input = F.elu(x + data.input)
        #data.input = self.bn7(data.input)
        data.batch = Variable(data.batch.view(-1, 1).expand(data.input.size()))
        x = scatter_mean(data.batch, data.input)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#show_pointclouds(single_example)


def train(epoch):
    model.train()

    if epoch == 11:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    if epoch == 21:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    example = 0
    loss_count = 0
    loss_sum = 0
    for i, data in enumerate(train_loader):
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.data[0]
        loss_count += 1
        if example % 100 == 0:
            print('Example', example, 'of', len(train_dataset), '. Loss:',
                  loss_sum / loss_count)
            loss_count = 0
            loss_sum = 0
        example += batch_size


def test(epoch, loader, ds, string):
    model.eval()
    correct = 0

    for data in loader:
        data = data.cuda().to_variable(['input'])
        output = model(data)
        pred = output.data.max(1)[1]
        #print(pred.size(),data.target.size())
        correct += pred.eq(data.target).cpu().sum()
    print(correct)
    print('Epoch:', epoch, string, 'Accuracy:', correct / len(ds))


for epoch in range(1, 31):
    train(epoch)
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, test_loader, test_dataset, 'Test')
