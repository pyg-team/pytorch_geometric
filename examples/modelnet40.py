import os
import sys

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
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import voxel_max_pool  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10')


transform = Compose([
    #RandomRotate(3.14),
    #RandomScale(1.1),
    #RandomShear(0.5),
    #RandomTranslate(0.005),
    NormalizeScale(),
    CartesianAdj()
])


trans = CartesianAdj()

batch_size = 32

train_dataset = ModelNet40(path, True, transform=transform)
test_dataset = ModelNet40(path, False, transform=transform)
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size, shuffle=True)
num_classes=40

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=3, kernel_size=15)
        self.conv2 = SplineConv(65, 64, dim=3, kernel_size=15)
        self.conv22 = SplineConv(64, 64, dim=3, kernel_size=15)
        self.conv3 = SplineConv(65, 64, dim=3, kernel_size=15)
        self.conv32 = SplineConv(64, 64, dim=3, kernel_size=15)
        self.conv4 = SplineConv(65, 64, dim=3, kernel_size=15)
        self.conv42 = SplineConv(64, 64, dim=3, kernel_size=15)
        self.conv5 = SplineConv(65, 64, dim=3, kernel_size=15)
        self.conv52 = SplineConv(64, 64, dim=3, kernel_size=15)
        self.conv6 = SplineConv(65, 64, dim=3, kernel_size=15)
        self.conv62 = SplineConv(64, 64, dim=3, kernel_size=15)
        self.conv7 = SplineConv(65, 64, dim=3, kernel_size=15)
        self.conv72 = SplineConv(64, 64, dim=3, kernel_size=15)
        #self.conv8 = SplineConv(256, 256, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(64, num_classes)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)

    def forward(self, data):
        ones = Variable(data.input.data.new(data.input.size(0), 1).fill_(1))
        #print('Num nodes', data.num_nodes)
        data.input = F.elu(self.conv1(data.adj, data.input))
        data.input = self.bn1(data.input)
        data, _ = voxel_max_pool(data,(1/128), origin=0, transform=trans)
        x = torch.cat([data.input,ones[:data.input.size(0)]],dim=1)
        x = F.elu(self.conv2(data.adj, x))
        x = self.conv22(data.adj, x)
        data.input = F.elu(x + data.input)
        data.input = self.bn2(data.input)
        data, _ = voxel_max_pool(data, (1/64), origin=0, transform=trans)
        x = torch.cat((data.input,ones[:data.input.size(0),:]),dim=1)
        x = F.elu(self.conv3(data.adj, x))
        x = self.conv32(data.adj, x)
        data.input = F.elu(x + data.input)
        data.input = self.bn3(data.input)
        data, _ = voxel_max_pool(data, (1/32), origin=0, transform=trans)
        x = torch.cat([data.input,ones[:data.input.size(0)]],dim=1)
        x = F.elu(self.conv4(data.adj, x))
        x = self.conv42(data.adj, x)
        data.input = F.elu(x + data.input)
        data.input = self.bn4(data.input)
        data, _ = voxel_max_pool(data, (1/16), origin=0, transform=trans)
        x = torch.cat([data.input,ones[:data.input.size(0)]],dim=1)
        x = F.elu(self.conv5(data.adj, x))
        x = self.conv52(data.adj, x)
        data.input = F.elu(x + data.input)
        data.input = self.bn5(data.input)
        data, _ = voxel_max_pool(data, (1/8), origin=0, transform=trans)
        x = torch.cat([data.input,ones[:data.input.size(0)]],dim=1)
        x = F.elu(self.conv6(data.adj, x))
        x = self.conv62(data.adj, x)
        data.input = F.elu(x + data.input)
        data.input = self.bn6(data.input)

        data, _ = voxel_max_pool(data, (1/4), origin=0, transform=trans)
        x = torch.cat([data.input, ones[:data.input.size(0)]], dim=1)
        x = F.elu(self.conv7(data.adj, x))
        x = self.conv72(data.adj, x)
        data.input = F.elu(x + data.input)
        data.input = self.bn7(data.input)
        data.batch = Variable(data.batch.view(-1, 1).expand(data.input.size()))
        x = scatter_mean(data.batch, data.input)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)


        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    #if epoch == 21:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.0001
    example = 0
    for i,data in enumerate(train_loader):
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.target)
        loss.backward()
        optimizer.step()
        print('Example',example, 'of',len(train_dataset),'. Loss:', loss.data[0])
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
    test(epoch,train_loader, train_dataset, 'Train')
    test(epoch,test_loader, test_dataset, 'Test')
